from typing import Dict, Optional

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import MlxPretrainedMixin
from .cache import Cache, DynamicCache
from .llama import LlamaRotaryEmbedding, apply_rotary_pos_emb, repeat_kv
from .modelling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from .utils import ACT2FN


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = mx.power(hidden_states, 2)
        variance = mx.mean(variance, axis=-1, keepdims=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = (
            config.layer_types[layer_idx]
            if hasattr(config, "layer_types") and config.layer_types is not None
            else "full_attention"
        )
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
    ):
        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).reshape(hidden_shape)
        key_states = self.k_proj(hidden_states).reshape(hidden_shape)
        value_states = self.v_proj(hidden_states).reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        query_states = self.q_norm(query_states).transpose(0, 2, 1, 3)
        key_states = self.k_norm(key_states).transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = (
            query_states.astype(mx.float32)
            @ key_states.astype(mx.float32).transpose(0, 1, 3, 2)
        ) * self.scaling

        if attention_mask is not None:
            attn_weights = (
                attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
            )

        attn_weights = mx.softmax(attn_weights, axis=-1).astype(query_states.dtype)
        attn_output = (attn_weights @ value_states).transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_values


class Qwen3SdpaAttention(Qwen3Attention):
    def __call__(
        self,
        hidden_states: mx.array,
        position_embeddings,
        attention_mask: Optional[mx.array] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ):
        if output_attentions or (self.training and self.attention_dropout > 0):
            return super().__call__(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.shape
        hidden_shape = (bsz, q_len, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).reshape(hidden_shape)
        key_states = self.k_proj(hidden_states).reshape(hidden_shape)
        value_states = self.v_proj(hidden_states).reshape(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        )

        query_states = self.q_norm(query_states).transpose(0, 2, 1, 3)
        key_states = self.k_norm(key_states).transpose(0, 2, 1, 3)
        value_states = value_states.transpose(0, 2, 1, 3)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx
            )

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_output = mx.fast.scaled_dot_product_attention(
            q=query_states,
            k=key_states,
            v=value_states,
            scale=self.scaling,
            mask=attention_mask[:, :, :, : key_states.shape[-2]]
            if attention_mask is not None
            else None,
        )

        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)
        attn_output = self.o_proj(attn_output)

        return attn_output, None, past_key_values


QWEN3_ATTENTION_CLASSES = {
    "eager": Qwen3Attention,
    "sdpa": Qwen3SdpaAttention,
}


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        attn_implementation = getattr(config, "_attn_implementation", None) or "eager"
        self.self_attn = QWEN3_ATTENTION_CLASSES[attn_implementation](
            config=config, layer_idx=layer_idx
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        use_cache=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )

        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        if use_cache:
            outputs += (present_key_value,)
        return outputs


class Qwen3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        layer_types = getattr(config, "layer_types", None)
        if layer_types is not None and any(
            layer_type != "full_attention" for layer_type in layer_types
        ):
            raise NotImplementedError(
                "Qwen3 sliding-window attention is not implemented yet."
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        self._attn_implementation = getattr(config, "_attn_implementation", "eager")

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        past_seen_tokens = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_usable_length(inputs_embeds.shape[1])

        if cache_position is None:
            cache_position = mx.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )

        if position_ids is None:
            position_ids = mx.expand_dims(cache_position, axis=0)

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_seen_tokens
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
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
        dtype = input_tensor.dtype
        min_dtype = np.finfo(np.float32).min
        sequence_length = input_tensor.shape[1]
        target_length = (
            attention_mask.shape[-1]
            if isinstance(attention_mask, mx.array)
            or isinstance(attention_mask, np.ndarray)
            else past_seen_tokens + sequence_length + 1
        )
        causal_mask = mx.full(
            (sequence_length, target_length), vals=min_dtype, dtype=dtype
        )
        if sequence_length != 1:
            causal_mask = mx.triu(causal_mask, k=1)

        causal_mask *= mx.arange(target_length) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :]
        causal_mask = mx.broadcast_to(
            causal_mask, (input_tensor.shape[0], 1, sequence_length, target_length)
        )

        if attention_mask is not None:
            if len(attention_mask.shape) == 2:
                mask_length = attention_mask.shape[-1]

                attention_mask = np.array(attention_mask)
                causal_mask = np.array(causal_mask)

                padding_mask = (causal_mask[..., :mask_length] == 0.0) * (
                    attention_mask[:, None, None, :] == 0.0
                )

                causal_mask[..., :mask_length] = np.ma.array(
                    data=causal_mask[..., :mask_length], mask=padding_mask
                ).filled(min_dtype)

            elif len(attention_mask.shape) == 4:
                mask_shape = attention_mask.shape
                mask_slice = (attention_mask == 0.0).astype(dtype) * min_dtype
                causal_mask[
                    : mask_shape[0],
                    : mask_shape[1],
                    : mask_shape[2],
                    : mask_shape[3],
                ] = mask_slice

        return mx.array(causal_mask)


class Qwen3ForCausalLM(nn.Module, MlxPretrainedMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def __call__(
        self,
        input_ids=None,
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
        num_logits_to_keep: int = 0,
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

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        if input_ids is None and inputs_embeds is None:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        current_input_length = (
            input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]
        )

        if attention_mask is not None and position_ids is None:
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
                position_ids = position_ids[:, -current_input_length:]

        outputs = self.model(
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
        logits_to_keep = (
            hidden_states
            if num_logits_to_keep == 0
            else hidden_states[:, -num_logits_to_keep:, :]
        )
        logits = self.lm_head(logits_to_keep).astype(mx.float32)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].astype(mx.int32)
            valid_mask = (shift_labels != -100).astype(shift_logits.dtype)
            safe_labels = mx.where(shift_labels != -100, shift_labels, 0)
            token_loss = nn.losses.cross_entropy(
                shift_logits,
                safe_labels,
                reduction="none",
            )
            loss = mx.sum(token_loss * valid_mask) / mx.maximum(mx.sum(valid_mask), 1.0)

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
                past_length = past_key_values.seen_tokens
            else:
                past_length = past_key_values[0][0].shape[2]

            if (
                attention_mask is not None
                and attention_mask.shape[1] > input_ids.shape[1]
            ):
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
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

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
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
            return mx.random.categorical(logits * (1 / temp))

        use_cache = kwargs.get("use_cache", True)
        output = self(**inputs, use_cache=use_cache)

        next_token_logits = output.logits[:, -1, :]
        next_token = sample(next_token_logits)

        yield next_token
        generated_tokens = 1

        while generated_tokens < max_length:
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
            generated_tokens += 1
