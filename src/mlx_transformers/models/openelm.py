import logging
from itertools import accumulate
from typing import Optional, Tuple, Dict, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from transformers import AutoConfig

from .base import MlxPretrainedMixin
from .cache import Cache, DynamicCache
from .modelling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .utils import ACT2FN

logger = logging.getLogger(__name__)


def make_divisible(
    v: Union[float, int],
    divisor: Optional[int] = 8,
    min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by the divisor
    It can be seen at:
    https://github.com/tensorflow/models/blob/2cfc99eff5e5eb729c6793d2f3d03aa1c9be2b15/research/slim/nets/mobilenet/mobilenet.py#L62
    Args:
        v: input value
        divisor: default to 8
        min_value: minimum divisor value
    Returns:
        new_v: new divisible value
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class OpenELMRMSNorm(nn.Module):
    def __init__(self, num_features, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.eps = eps
        self.num_features = num_features

    def __call__(self, x):
        input_dtype = x.dtype
        x = x.astype(mx.float32)
        variance = mx.power(x, 2)
        variance = mx.mean(variance, axis=-1, keepdims=True)
        x = x * mx.rsqrt(variance + self.eps)
        return self.weight * x.astype(input_dtype)


def rotate_half(x):
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


def _apply_rotary_pos_emb(x, pos_sin, pos_cos):
    return (x * pos_cos) + (rotate_half(x) * pos_sin)


class OpenELMRotaryEmbedding(nn.Module):
    def __init__(self, model_dim, max_seq_length=2048, freq_constant=10000):
        super().__init__()

        self.inv_freq = 1.0 / (
            freq_constant ** (mx.arange(0, model_dim, 2) / model_dim)
        )
        self.model_dim = model_dim
        self.max_seq_length = max_seq_length
        self.freq_constant = freq_constant

        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_length = max_seq_length
        self._compute_sin_cos_embeddings(max_seq_length)

    def _compute_sin_cos_embeddings(
        self,
        key_len: int,
        key_dtype: mx.Dtype = mx.float32,
    ) -> None:
        if (
            key_len > self._cached_seq_length
            or self._cached_cos is None
            or self._cached_sin is None
        ):
            self._cached_seq_length = max(key_len, self._cached_seq_length)
            pos_index = mx.arange(self._cached_seq_length)

            pos_index_theta = np.einsum(
                "i,j->ij", np.array(pos_index), np.array(self.inv_freq)
            )
            pos_index_theta = mx.array(pos_index_theta)

            emb = mx.concatenate((pos_index_theta, pos_index_theta), axis=-1)

            cos_emb = mx.cos(emb)
            sin_emb = mx.sin(emb)

            self._cached_cos = cos_emb[None, None, :, :]
            self._cached_sin = sin_emb[None, None, :, :]

    def __call__(self, query, key):
        dim = key.shape[-1]
        key_len = key.shape[2]
        query_len = query.shape[2]

        assert dim == self.model_dim
        assert key.dtype == query.dtype

        query_float = query.astype(mx.float32)
        key_float = key.astype(mx.float32)

        self._compute_sin_cos_embeddings(key_len, key_float.dtype)

        query_float = _apply_rotary_pos_emb(
            x=query_float,
            pos_sin=self._cached_sin[..., key_len - query_len : key_len, :],
            pos_cos=self._cached_cos[..., key_len - query_len : key_len, :],
        )
        key_float = _apply_rotary_pos_emb(
            x=key_float,
            pos_sin=self._cached_sin[..., :key_len, :],
            pos_cos=self._cached_cos[..., :key_len, :],
        )

        return query_float.astype(query.dtype), key_float.astype(key.dtype)


class OpenELMMultiHeadCausalAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: AutoConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        head_dim = config.head_dim
        q_heads = config.num_query_heads[layer_idx]
        k_heads = config.num_kv_heads[layer_idx]
        v_heads = config.num_kv_heads[layer_idx]

        self.qkv_proj = nn.Linear(
            config.model_dim,
            (q_heads + k_heads + v_heads) * head_dim,
            bias=False,
        )

        self.pos_embedding = OpenELMRotaryEmbedding(
            model_dim=config.head_dim,
            max_seq_length=config.rope_max_length,
            freq_constant=config.rope_freq_constant,
        )

        if config.normalize_qk_projections:
            self.q_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
            self.k_norm = OpenELMRMSNorm(
                num_features=config.head_dim,
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.out_proj = nn.Linear(
            q_heads * head_dim,
            config.model_dim,
            bias=False,
        )

        self.head_dim = config.head_dim
        self.num_q_heads = q_heads
        self.num_k_heads = k_heads
        self.num_v_heads = v_heads
        self.transformer_dim = config.model_dim
        self.num_groups = self.num_q_heads // self.num_k_heads

        self.scale = self.head_dim**-0.5

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=False,
        use_cache=False,
        cache_position=None,
    ):
        batch_size, seq_length, d_model = hidden_states.shape

        qkv = self.qkv_proj(hidden_states)

        qkv = qkv.reshape(
            batch_size,
            seq_length,
            self.num_q_heads + self.num_k_heads + self.num_v_heads,
            self.head_dim,
        )

        qkv = qkv.transpose(0, 2, 1, 3)

        index = list(accumulate([self.num_q_heads, self.num_k_heads, self.num_v_heads]))

        queries, keys, values = mx.split(qkv, index[:-1], axis=1)

        if self.q_norm is not None:
            queries = self.q_norm(queries)

        if self.k_norm is not None:
            keys = self.k_norm(keys)

        if past_key_value is not None:
            cache_kwargs = {"cache_position": cache_position}
            keys, values = past_key_value.update(
                keys, values, self.layer_idx, cache_kwargs
            )

        queries, keys = self.pos_embedding(queries, keys)

        if self.num_groups != 1:
            keys = mx.repeat(keys, repeats=self.num_groups, axis=1)
            values = mx.repeat(values, repeats=self.num_groups, axis=1)

        causal_mask = attention_mask
        if attention_mask is not None and cache_position is not None:
            causal_mask = causal_mask[:, :, cache_position, : keys.shape[-2]]

        attn_output = mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            mask=mx.array(causal_mask),
            scale=self.scale,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3)

        attn_output = attn_output.reshape(
            batch_size, seq_length, self.num_q_heads * self.head_dim
        )
        attn_output = self.out_proj(attn_output)
        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights, past_key_value


class OpenELMFeedForwardNetwork(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        ffn_multiplier = config.ffn_multipliers[layer_idx]
        intermediate_dim = int(
            make_divisible(
                ffn_multiplier * config.model_dim,
                divisor=config.ffn_dim_divisor,
            )
        )
        if config.ffn_with_glu:
            # FFN with Gated linear unit, as described in https://arxiv.org/abs/2002.05202v1.
            self.proj_1 = nn.Linear(
                config.model_dim,
                2 * intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                intermediate_dim,
                config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = True
        else:
            # Standard FFN, as described in https://arxiv.org/abs/1706.03762
            self.proj_1 = nn.Linear(
                config.model_dim,
                intermediate_dim,
                bias=False,
            )
            self.proj_2 = nn.Linear(
                intermediate_dim,
                config.model_dim,
                bias=False,
            )
            self.ffn_with_glu = False

        self.act = ACT2FN[config.activation_fn_name]

    def __call__(self, x: mx.array) -> mx.array:
        if self.ffn_with_glu:
            y_12 = self.proj_1(x)
            y_1, y_2 = mx.split(y_12, 2, axis=-1)
            y = self.act(y_1) * y_2
            y = self.proj_2(y)
        else:
            y = self.proj_1(x)
            y = self.act(y)
            y = self.proj_2(y)

        return y


class OpenELMDecoderLayer(nn.Module):
    def __init__(self, config: AutoConfig, layer_idx: int) -> None:
        super().__init__()
        self.attn = OpenELMMultiHeadCausalAttention(config=config, layer_idx=layer_idx)
        self.ffn = OpenELMFeedForwardNetwork(config=config, layer_idx=layer_idx)
        self.ffn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )
        self.attn_norm = OpenELMRMSNorm(
            num_features=config.model_dim,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: Optional[mx.array] = None,
        position_ids: Optional[mx.array] = None,
        past_key_value: Optional[Tuple[mx.array]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[mx.array] = None,
        **kwargs,
    ) -> Tuple[mx.array, Optional[Tuple[mx.array, mx.array]]]:
        residual = hidden_states
        hidden_states = self.attn_norm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            position_ids=position_ids,
            cache_position=cache_position,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.ffn(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class OpenELMModel(nn.Module):
    def __init__(self, config: AutoConfig):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(config.model_dim, config.vocab_size)

        self.layers = [
            OpenELMDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_transformer_layers)
        ]
        self.norm = OpenELMRMSNorm(num_features=config.model_dim)
        if config.share_input_output_layers:
            self.classifier = None
        else:
            self.classifier = nn.Linear(
                config.model_dim,
                config.vocab_size,
                bias=False,
            )
        self.num_transformer_layers = config.num_transformer_layers

        causal_mask = mx.full(
            (config.max_context_length, config.max_context_length),
            vals=True,
            dtype=mx.bool_,
        )
        self.causal_mask = mx.triu(causal_mask, 1)

    def get_input_embeddings(self):
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings):
        self.token_embeddings = new_embeddings

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape[:2]
        elif inputs_embeds is not None:
            batch_size, seq_length = inputs_embeds.shape[:2]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_usable_length(seq_length)

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        if cache_position is None:
            cache_position = mx.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )

        if position_ids is None:
            position_ids = mx.expand_dims(cache_position, axis=0)

        # TODO: implement cache
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, position_ids, past_seen_tokens
        )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None

        if use_cache:
            next_cache = (
                next_decoder_cache.to_legacy_cache()
                if isinstance(next_decoder_cache, Cache)
                else next_decoder_cache
            )

        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
                if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: mx.array,
        input_tensor: mx.array,
        cache_position: mx.array,
        past_seen_tokens: int,
    ):
        batch_size, seq_length = input_tensor.shape[:2]
        dtype = input_tensor.dtype

        if seq_length > self.causal_mask.shape[-1]:
            causal_mask = mx.full(
                (2 * self.causal_mask.shape[-1], 2 * self.causal_mask.shape[-1]), vals=1
            )
            self.causal_mask = mx.triu(causal_mask, 1)

        min_dtype = np.finfo(np.float32).min

        causal_mask = self.causal_mask[None, None, :, :]
        causal_mask = (
            mx.tile(causal_mask, (batch_size, 1, 1, 1)).astype(dtype) * min_dtype
        )
        causal_mask = causal_mask.astype(dtype)

        if attention_mask is not None and len(attention_mask.shape) == 2:
            mask_length = attention_mask.shape[-1]
            attention_mask = np.array(attention_mask)
            causal_mask = np.array(causal_mask)

            padding_mask = (causal_mask[..., :mask_length] == 0.0) * (
                attention_mask[:, None, None, :] == 0.0
            )

            causal_mask[..., :mask_length] = np.ma.array(
                data=causal_mask[..., :mask_length], mask=padding_mask
            ).filled(min_dtype)

        return causal_mask


class OpenELMForCausalLM(nn.Module, MlxPretrainedMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = OpenELMModel(config)
        self.vocab_size = config.vocab_size
        if config.share_input_output_layers:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.transformer.token_embeddings

    def set_input_embeddings(self, value):
        self.transformer.token_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.transformer = decoder

    def get_decoder(self):
        return self.transformer

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state

        if self.lm_head is None:
            # shared
            logits = hidden_states @ mx.transpose(
                self.transformer.token_embeddings.weight, (1, 0)
            )
        else:
            logits = self.lm_head(hidden_states)

        logits = logits[:, : self.config.vocab_size]
        loss = None

        if labels is not None:
            # TODO: implement loss
            pass

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = past_key_values.get_max_length()
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids,
            # then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g.
            # when passing input_embeds as input)
            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds
            # all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids
            # only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the
            # input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.astype(mx.int32).cumsum(-1) - 1

            position_ids, attention_mask = (
                np.array(position_ids),
                np.array(attention_mask),
            )

            position_ids = np.ma.array(
                data=position_ids, mask=attention_mask == 0
            ).filled(1)
            position_ids = mx.array(position_ids)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during
            # decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref:
            # https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids}

        input_length = (
            position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        )

        if cache_position is None:
            cache_position = mx.arange(past_length, past_length + input_length)
        elif use_cache:
            cache_position = cache_position[-input_length:]

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )

        return model_inputs

    def generate(self, inputs: Dict, max_length: int, **kwargs):
        temp = kwargs.get("temp", 1.0)

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            else:
                return mx.random.categorical(logits * (1 / temp))

        # Process the prompt
        inputs = self.prepare_inputs_for_generation(
            input_ids=inputs["input_ids"],
            past_key_values=None,
            attention_mask=inputs["attention_mask"],
            inputs_embeds=None,
        )
        output = self(**inputs)
        next_token_logits = output.logits[:, -1, :]
        next_token = sample(next_token_logits)

        yield next_token

        while True:
            next_token = mx.expand_dims(next_token, axis=0)

            inputs["input_ids"] = next_token
            inputs["attention_mask"] = mx.concatenate(
                [mx.array(inputs["attention_mask"]), mx.ones_like(next_token)], axis=-1
            )

            past_key_values = output.past_key_values
            inputs = self.prepare_inputs_for_generation(
                input_ids=inputs["input_ids"],
                past_key_values=past_key_values,
                attention_mask=inputs["attention_mask"],
                inputs_embeds=None,
            )

            output = self(**inputs)

            next_token_logits = output.logits[:, -1, :]
            next_token = sample(next_token_logits)
            yield next_token
