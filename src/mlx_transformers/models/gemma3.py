from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .base import MlxPretrainedMixin
from .cache import Cache, DynamicCache
from .modelling_outputs import BaseModelOutputWithPast
from .utils import ACT2FN


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states

    hidden_states = hidden_states[:, :, None, :, :]
    hidden_states = mx.broadcast_to(
        hidden_states, (batch, num_key_value_heads, n_rep, seq_len, head_dim)
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, seq_len, head_dim)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = mx.expand_dims(cos, axis=unsqueeze_dim)
    sin = mx.expand_dims(sin, axis=unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


@dataclass
class BaseModelOutputWithPooling:
    last_hidden_state: mx.array = None
    pooler_output: mx.array = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None


@dataclass
class Gemma3ModelOutputWithPast:
    last_hidden_state: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    image_hidden_states: Optional[mx.array] = None


@dataclass
class Gemma3CausalLMOutputWithPast:
    loss: Optional[mx.array] = None
    logits: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    image_hidden_states: Optional[mx.array] = None


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        patch_area = config.patch_size * config.patch_size * config.num_channels
        self.patch_embedding = nn.Linear(patch_area, self.embed_dim)
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.position_ids = mx.arange(self.num_positions)[None, :]

    def __call__(self, pixel_values):
        batch_size, channels, height, width = pixel_values.shape
        patch_size = self.patch_size

        if height != self.image_size or width != self.image_size:
            raise ValueError(
                "SigLIP interpolation is not implemented in this MLX port. "
                f"Expected {self.image_size}x{self.image_size}, got {height}x{width}."
            )

        pixels = np.array(pixel_values)
        patches = pixels.reshape(
            batch_size,
            channels,
            height // patch_size,
            patch_size,
            width // patch_size,
            patch_size,
        )
        patches = patches.transpose(0, 2, 4, 1, 3, 5).reshape(
            batch_size, -1, channels * patch_size * patch_size
        )
        patches = mx.array(patches)
        embeddings = self.patch_embedding(
            patches.astype(self.patch_embedding.weight.dtype)
        )
        return embeddings + self.position_embedding(self.position_ids)


class SiglipAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def __call__(self, hidden_states, attention_mask=None):
        batch_size, seq_length, embed_dim = hidden_states.shape
        queries = self.q_proj(hidden_states).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        keys = self.k_proj(hidden_states).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )
        values = self.v_proj(hidden_states).reshape(
            batch_size, seq_length, self.num_heads, self.head_dim
        )

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        attn_weights = (
            queries.astype(mx.float32) @ keys.astype(mx.float32).transpose(0, 1, 3, 2)
        ) * self.scale
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = mx.softmax(attn_weights, axis=-1).astype(queries.dtype)
        attn_output = attn_weights @ values
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_length, embed_dim
        )
        return self.out_proj(attn_output), attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def __call__(self, hidden_states):
        return self.fc2(self.activation_fn(self.fc1(hidden_states)))


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.self_attn = SiglipAttention(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)

    def __call__(self, hidden_states, attention_mask=None):
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = [
            SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)
        ]

    def __call__(self, inputs_embeds, attention_mask=None):
        hidden_states = inputs_embeds
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states, attention_mask)
        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def __call__(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return BaseModelOutputWithPooling(last_hidden_state=hidden_states)


class SiglipVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vision_model = SiglipVisionTransformer(config)

    def get_input_embeddings(self):
        return self.vision_model.embeddings.patch_embedding

    def __call__(self, pixel_values):
        return self.vision_model(pixel_values)


class Gemma3TextScaledWordEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        embed_scale: float = 1.0,
    ):
        super().__init__(num_embeddings, embedding_dim)
        self.padding_idx = padding_idx
        self.scalar_embed_scale = embed_scale
        self.embed_scale = mx.array(embed_scale)

    def __call__(self, input_ids):
        return super().__call__(input_ids) * self.embed_scale.astype(self.weight.dtype)


class Gemma3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def __call__(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.zeros((dim,))

    def __call__(self, x):
        output = x.astype(mx.float32)
        output = output * mx.rsqrt(
            mx.mean(mx.power(output, 2), axis=-1, keepdims=True) + self.eps
        )
        output = output * (1.0 + self.weight.astype(mx.float32))
        return output.astype(x.dtype)


class Gemma3RotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_types = list(dict.fromkeys(config.layer_types))
        self.inv_freq = {}
        self.attention_scaling = {}
        for layer_type in self.layer_types:
            rope_params = config.rope_parameters[layer_type]
            base = rope_params["rope_theta"]
            dim = getattr(
                config, "head_dim", config.hidden_size // config.num_attention_heads
            )
            self.inv_freq[layer_type] = 1.0 / (
                base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim)
            )
            self.attention_scaling[layer_type] = 1.0

    def __call__(self, x, position_ids, layer_type=None):
        inv_freq = self.inv_freq[layer_type]
        attention_scaling = self.attention_scaling[layer_type]
        inv_freq_expanded = inv_freq[None, :, None].astype(mx.float32)
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_expanded, (position_ids.shape[0], inv_freq_expanded.shape[1], 1)
        )
        position_ids_expanded = position_ids[:, None, :].astype(mx.float32)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 2, 1)
        emb = mx.concatenate((freqs, freqs), axis=-1)
        cos = mx.cos(emb) * attention_scaling
        sin = mx.sin(emb) * attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


class Gemma3Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_type = config.layer_types[layer_idx]
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = config.query_pre_attn_scalar**-0.5
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.attn_logit_softcapping = self.config.attn_logit_softcapping
        self.q_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Gemma3RMSNorm(dim=config.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        output_attentions=False,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = (
            self.q_proj(hidden_states).reshape(hidden_shape).transpose(0, 2, 1, 3)
        )
        key_states = (
            self.k_proj(hidden_states).reshape(hidden_shape).transpose(0, 2, 1, 3)
        )
        value_states = self.v_proj(hidden_states).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_key_value_heads,
            self.head_dim,
        )
        value_states = value_states.transpose(0, 2, 1, 3)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

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

        if self.attn_logit_softcapping is not None:
            attn_weights = attn_weights / self.attn_logit_softcapping
            attn_weights = mx.tanh(attn_weights)
            attn_weights = attn_weights * self.attn_logit_softcapping

        if attention_mask is not None:
            attn_weights = (
                attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
            )

        attn_weights = mx.softmax(attn_weights, axis=-1).astype(query_states.dtype)
        attn_output = (attn_weights @ value_states).transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(*input_shape, -1)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None
        return attn_output, attn_weights


class Gemma3DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.self_attn = Gemma3Attention(config=config, layer_idx=layer_idx)
        self.mlp = Gemma3MLP(config)
        self.input_layernorm = Gemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma3RMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        past_key_values=None,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_weights,)
        outputs += (past_key_values,)
        return outputs


def _compute_image_group_ids(token_type_ids):
    token_type_ids_np = np.array(token_type_ids).astype(np.int32)
    is_image = token_type_ids_np == 1
    is_previous_image = np.pad(is_image, ((0, 0), (1, 0)), constant_values=False)[
        :, :-1
    ]
    new_image_start = is_image & ~is_previous_image
    image_group_ids = np.cumsum(new_image_start.astype(np.int32), axis=1) - 1
    image_group_ids = np.where(is_image, image_group_ids, -1)
    return image_group_ids


def _apply_image_bidirectional_overlay(causal_mask, token_type_ids):
    if token_type_ids is None:
        return causal_mask

    mask_np = np.array(causal_mask)
    token_type_ids_np = np.array(token_type_ids)
    image_group_ids = _compute_image_group_ids(token_type_ids_np)

    for batch_idx in range(token_type_ids_np.shape[0]):
        batch_group_ids = image_group_ids[batch_idx]
        image_positions = np.nonzero(batch_group_ids >= 0)[0]
        for q_idx in image_positions:
            same_group = image_positions[
                batch_group_ids[image_positions] == batch_group_ids[q_idx]
            ]
            mask_np[batch_idx, 0, q_idx, same_group] = 0.0

    return mx.array(mask_np)


def _make_causal_mask(
    attention_mask,
    input_tensor,
    past_seen_tokens,
    sliding_window=None,
    token_type_ids=None,
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

    cache_position = mx.arange(past_seen_tokens, past_seen_tokens + sequence_length)
    query_positions = cache_position.reshape(-1, 1)
    key_positions = mx.arange(target_length).reshape(1, -1)

    causal_allowed = key_positions <= query_positions
    if sliding_window is not None:
        causal_allowed = causal_allowed & (
            key_positions >= (query_positions - sliding_window + 1)
        )

    causal_mask = np.full((sequence_length, target_length), min_dtype, dtype=np.float32)
    causal_mask[causal_allowed] = 0.0
    causal_mask = causal_mask[None, None, :, :]
    causal_mask = np.broadcast_to(
        causal_mask, (input_tensor.shape[0], 1, sequence_length, target_length)
    ).copy()

    if attention_mask is not None:
        if len(attention_mask.shape) == 2:
            attention_mask_np = np.array(attention_mask)
            mask_length = attention_mask_np.shape[-1]
            padding_mask = (causal_mask[..., :mask_length] == 0.0) & (
                attention_mask_np[:, None, None, :] == 0.0
            )
            causal_mask[..., :mask_length] = np.ma.array(
                data=causal_mask[..., :mask_length],
                mask=padding_mask,
            ).filled(min_dtype)
        elif len(attention_mask.shape) == 4:
            attention_mask_np = np.array(attention_mask)
            mask_shape = attention_mask_np.shape
            mask_slice = (attention_mask_np == 0.0).astype(np.float32) * min_dtype
            causal_mask[
                : mask_shape[0],
                : mask_shape[1],
                : mask_shape[2],
                : mask_shape[3],
            ] = mask_slice

    causal_mask = mx.array(causal_mask).astype(dtype)
    if token_type_ids is not None and past_seen_tokens == 0:
        causal_mask = _apply_image_bidirectional_overlay(causal_mask, token_type_ids)
    return causal_mask


def create_causal_mask_mapping(
    config,
    inputs_embeds,
    attention_mask,
    past_key_values,
    position_ids,
    token_type_ids=None,
    pixel_values=None,
):
    text_config = getattr(config, "text_config", config)
    past_seen_tokens = (
        0 if past_key_values is None else past_key_values.get_seq_length()
    )
    full_attention_mask = _make_causal_mask(
        attention_mask=attention_mask,
        input_tensor=inputs_embeds,
        past_seen_tokens=past_seen_tokens,
        sliding_window=None,
        token_type_ids=token_type_ids if pixel_values is not None else None,
    )
    sliding_attention_mask = _make_causal_mask(
        attention_mask=attention_mask,
        input_tensor=inputs_embeds,
        past_seen_tokens=past_seen_tokens,
        sliding_window=text_config.sliding_window,
        token_type_ids=token_type_ids if pixel_values is not None else None,
    )
    return {
        "full_attention": full_attention_mask,
        "sliding_attention": sliding_attention_mask,
    }


class Gemma3TextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = Gemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=self.config.hidden_size**0.5,
        )
        self.layers = [
            Gemma3DecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma3RotaryEmbedding(config)

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
    ):
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else getattr(self.config, "output_attentions", False)
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "use_return_dict", True)
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        else:
            past_key_values = None

        if position_ids is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            position_ids = mx.arange(inputs_embeds.shape[1]) + past_seen_tokens
            position_ids = mx.expand_dims(position_ids, axis=0)

        causal_mask_mapping = attention_mask
        hidden_states = inputs_embeds
        position_embeddings = {}
        for layer_type in dict.fromkeys(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(
                hidden_states,
                position_ids,
                layer_type,
            )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings[self.config.layer_types[i]],
                past_key_values=past_key_values,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache and past_key_values is not None:
            next_cache = past_key_values.to_legacy_cache()

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


class Gemma3ForCausalLM(nn.Module, MlxPretrainedMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gemma3TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

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
        logits_to_keep: int = 0,
    ):
        if not isinstance(attention_mask, dict):
            attention_mask = create_causal_mask_mapping(
                self.config,
                inputs_embeds
                if inputs_embeds is not None
                else self.get_input_embeddings()(input_ids),
                attention_mask,
                DynamicCache.from_legacy_cache(past_key_values)
                if past_key_values is not None
                and not isinstance(past_key_values, Cache)
                else past_key_values,
                position_ids,
            )

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
        )
        hidden_states = outputs.last_hidden_state
        logits_slice = (
            hidden_states
            if logits_to_keep == 0
            else hidden_states[:, -logits_to_keep:, :]
        )
        logits = self.lm_head(logits_slice)
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = mx.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
        logits = logits.astype(mx.float32)

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

        return Gemma3CausalLMOutputWithPast(
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
        inputs_embeds=None,
        position_ids=None,
        attention_mask=None,
        use_cache=True,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]

        return {
            "input_ids": input_ids
            if inputs_embeds is None or past_key_values is not None
            else None,
            "inputs_embeds": inputs_embeds if past_key_values is None else None,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
        }

    def generate(self, inputs: Dict, max_length: int, **kwargs):
        temp = kwargs.get("temp", 1.0)
        use_cache = kwargs.get("use_cache", True)

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            return mx.random.categorical(logits * (1 / temp))

        model_inputs = self.prepare_inputs_for_generation(
            **inputs,
            use_cache=use_cache,
        )
        output = self(**model_inputs)
        next_token_logits = output.logits[:, -1, :]
        next_token = sample(next_token_logits)

        yield next_token
        generated_tokens = 1

        while generated_tokens < max_length:
            next_token = mx.expand_dims(next_token, axis=0)
            inputs["input_ids"] = next_token
            inputs["attention_mask"] = mx.concatenate(
                [mx.array(inputs["attention_mask"]), mx.ones_like(next_token)],
                axis=-1,
            )

            model_inputs = self.prepare_inputs_for_generation(
                input_ids=inputs["input_ids"],
                past_key_values=output.past_key_values,
                inputs_embeds=None,
                position_ids=None,
                attention_mask=inputs["attention_mask"],
                use_cache=use_cache,
            )
            output = self(**model_inputs)
            next_token_logits = output.logits[:, -1, :]
            next_token = sample(next_token_logits)
            yield next_token
            generated_tokens += 1


class Gemma3MultiModalProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mm_input_projection_weight = mx.zeros(
            (config.vision_config.hidden_size, config.text_config.hidden_size)
        )
        self.mm_soft_emb_norm = Gemma3RMSNorm(
            config.vision_config.hidden_size,
            eps=config.vision_config.layer_norm_eps,
        )
        self.patches_per_image = int(
            config.vision_config.image_size // config.vision_config.patch_size
        )
        self.tokens_per_side = int(config.mm_tokens_per_image**0.5)
        self.kernel_size = self.patches_per_image // self.tokens_per_side

    def __call__(self, vision_outputs):
        batch_size, _, hidden_size = vision_outputs.shape
        vision_np = np.array(vision_outputs)
        vision_np = vision_np.transpose(0, 2, 1).reshape(
            batch_size,
            hidden_size,
            self.patches_per_image,
            self.patches_per_image,
        )

        pooled = []
        for batch_idx in range(batch_size):
            sample = vision_np[batch_idx]
            sample = sample.reshape(
                hidden_size,
                self.tokens_per_side,
                self.kernel_size,
                self.tokens_per_side,
                self.kernel_size,
            ).mean(axis=(2, 4))
            pooled.append(sample.reshape(hidden_size, -1).transpose(1, 0))

        pooled = mx.array(np.stack(pooled, axis=0))
        pooled = self.mm_soft_emb_norm(pooled)
        projected = pooled @ self.mm_input_projection_weight
        return projected.astype(vision_outputs.dtype)


class Gemma3Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = Gemma3MultiModalProjector(config)
        self.vocab_size = config.text_config.vocab_size
        self.language_model = Gemma3TextModel(config.text_config)

    @property
    def image_token_id(self):
        return getattr(
            self.config,
            "image_token_id",
            getattr(self.config, "image_token_index"),
        )

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_image_features(self, pixel_values):
        vision_outputs = self.vision_tower(pixel_values)
        last_hidden_state = vision_outputs.last_hidden_state
        pooler_output = self.multi_modal_projector(last_hidden_state)
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooler_output,
        )

    def get_placeholder_mask(self, input_ids, inputs_embeds, image_features):
        if input_ids is None:
            image_token_embed = np.array(
                self.get_input_embeddings()(
                    mx.array([self.image_token_id], dtype=mx.int32)
                )
            )[0]
            special_image_mask = np.all(
                np.array(inputs_embeds) == image_token_embed, axis=-1
            )
        else:
            special_image_mask = np.array(input_ids) == self.image_token_id

        n_image_tokens = int(special_image_mask.sum())
        n_image_features = image_features.shape[0] * image_features.shape[1]
        if n_image_tokens != n_image_features:
            raise ValueError(
                "Image features and image tokens do not match, "
                f"tokens: {n_image_tokens}, features: {n_image_features}"
            )

        return mx.array(
            np.broadcast_to(special_image_mask[..., None], inputs_embeds.shape)
        )

    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )

        if input_ids is not None and self.image_token_id >= self.vocab_size:
            input_ids_np = np.array(input_ids).copy()
            input_ids_np[input_ids_np == self.image_token_id] = 0
            llm_input_ids = mx.array(input_ids_np)
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values).pooler_output
            image_features = image_features.astype(inputs_embeds.dtype)
            special_image_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_features,
            )
            inputs_embeds_np = np.array(inputs_embeds)
            image_mask_np = np.array(special_image_mask[..., 0]).astype(bool)
            inputs_embeds_np[image_mask_np] = np.array(image_features).reshape(
                -1, inputs_embeds_np.shape[-1]
            )
            inputs_embeds = mx.array(inputs_embeds_np)

        causal_mask_mapping = attention_mask
        if not isinstance(causal_mask_mapping, dict):
            causal_mask_mapping = create_causal_mask_mapping(
                self.config,
                inputs_embeds,
                attention_mask,
                DynamicCache.from_legacy_cache(past_key_values)
                if past_key_values is not None
                and not isinstance(past_key_values, Cache)
                else past_key_values,
                position_ids,
                token_type_ids,
                pixel_values,
            )

        outputs = self.language_model(
            attention_mask=causal_mask_mapping,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True if return_dict is None else return_dict,
        )

        return Gemma3ModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
        )


class Gemma3ForConditionalGeneration(nn.Module, MlxPretrainedMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Gemma3Model(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )

    def _apply_pretrained_tensors(self, tensors: Dict[str, mx.array]) -> None:
        tensors = dict(tensors)
        patch_embed_weight_key = (
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight"
        )
        if (
            patch_embed_weight_key in tensors
            and tensors[patch_embed_weight_key].ndim == 4
        ):
            tensors[patch_embed_weight_key] = tensors[patch_embed_weight_key].reshape(
                tensors[patch_embed_weight_key].shape[0], -1
            )
        MlxPretrainedMixin._apply_pretrained_tensors(self, tensors)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_image_features(self, pixel_values):
        return self.model.get_image_features(pixel_values)

    def __call__(
        self,
        input_ids=None,
        pixel_values=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        token_type_ids=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep: int = 0,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs.last_hidden_state
        logits_slice = (
            hidden_states
            if logits_to_keep == 0
            else hidden_states[:, -logits_to_keep:, :]
        )
        logits = self.lm_head(logits_slice)
        if self.config.text_config.final_logit_softcapping is not None:
            logits = logits / self.config.text_config.final_logit_softcapping
            logits = mx.tanh(logits)
            logits = logits * self.config.text_config.final_logit_softcapping
        logits = logits.astype(mx.float32)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None and not isinstance(attention_mask, dict):
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :]
                shift_logits = shift_logits[np.array(shift_attention_mask) != 0]
                shift_labels = shift_labels[np.array(shift_attention_mask) != 0]
            valid_mask = (shift_labels != -100).astype(shift_logits.dtype)
            safe_labels = mx.where(
                shift_labels != -100, shift_labels.astype(mx.int32), 0
            )
            token_loss = nn.losses.cross_entropy(
                shift_logits,
                safe_labels,
                reduction="none",
            )
            loss = mx.sum(token_loss * valid_mask) / mx.maximum(mx.sum(valid_mask), 1.0)

        return Gemma3CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        is_first_iteration=False,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]

        model_inputs = {
            "input_ids": input_ids
            if inputs_embeds is None or past_key_values is not None
            else None,
            "inputs_embeds": inputs_embeds if past_key_values is None else None,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }
        if is_first_iteration or not use_cache:
            model_inputs["pixel_values"] = pixel_values
        return model_inputs

    def generate(self, inputs: Dict, max_length: int, **kwargs):
        temp = kwargs.get("temp", 1.0)
        has_multimodal_inputs = inputs.get("pixel_values") is not None

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            return mx.random.categorical(logits * (1 / temp))

        use_cache = kwargs.get("use_cache", True) and not has_multimodal_inputs

        if not use_cache:
            generated_tokens = 0
            while generated_tokens < max_length:
                output = self(**inputs, use_cache=False)
                next_token_logits = output.logits[:, -1, :]
                next_token = sample(next_token_logits)
                yield next_token
                generated_tokens += 1

                next_token = mx.expand_dims(next_token, axis=0)
                inputs["input_ids"] = mx.concatenate(
                    [mx.array(inputs["input_ids"]), next_token], axis=-1
                )
                inputs["attention_mask"] = mx.concatenate(
                    [mx.array(inputs["attention_mask"]), mx.ones_like(next_token)],
                    axis=-1,
                )
                if inputs.get("token_type_ids") is not None:
                    inputs["token_type_ids"] = mx.concatenate(
                        [mx.array(inputs["token_type_ids"]), mx.zeros_like(next_token)],
                        axis=-1,
                    )
            return

        model_inputs = self.prepare_inputs_for_generation(
            **inputs,
            use_cache=use_cache,
            is_first_iteration=True,
        )
        output = self(**model_inputs)
        next_token_logits = output.logits[:, -1, :]
        next_token = sample(next_token_logits)

        yield next_token
        generated_tokens = 1

        while generated_tokens < max_length:
            next_token = mx.expand_dims(next_token, axis=0)
            inputs["input_ids"] = next_token
            inputs["attention_mask"] = mx.concatenate(
                [mx.array(inputs["attention_mask"]), mx.ones_like(next_token)],
                axis=-1,
            )
            if inputs.get("token_type_ids") is not None:
                inputs["token_type_ids"] = mx.concatenate(
                    [mx.array(inputs["token_type_ids"]), mx.zeros_like(next_token)],
                    axis=-1,
                )

            model_inputs = self.prepare_inputs_for_generation(
                input_ids=inputs["input_ids"],
                past_key_values=output.past_key_values,
                inputs_embeds=None,
                position_ids=None,
                pixel_values=inputs.get("pixel_values"),
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                use_cache=use_cache,
                is_first_iteration=False,
            )
            output = self(**model_inputs)
            next_token_logits = output.logits[:, -1, :]
            next_token = sample(next_token_logits)
            yield next_token
            generated_tokens += 1
