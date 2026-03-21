import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    q_dtype = q.dtype
    k_dtype = k.dtype
    cos = mx.expand_dims(cos.astype(mx.float32), axis=-2)
    sin = mx.expand_dims(sin.astype(mx.float32), axis=-2)
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.astype(q_dtype), k_embed.astype(k_dtype)


@dataclass
class BaseModelOutputWithDeepstackFeatures:
    last_hidden_state: mx.array = None
    pooler_output: mx.array = None
    deepstack_features: Optional[List[mx.array]] = None


@dataclass
class Qwen3VLModelOutputWithPast:
    last_hidden_state: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    rope_deltas: Optional[mx.array] = None


@dataclass
class Qwen3VLCausalLMOutputWithPast:
    loss: Optional[mx.array] = None
    logits: mx.array = None
    past_key_values: Optional[Tuple[Tuple[mx.array]]] = None
    hidden_states: Optional[Tuple[mx.array, ...]] = None
    attentions: Optional[Tuple[mx.array, ...]] = None
    rope_deltas: Optional[mx.array] = None


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.linear_fc1 = nn.Linear(self.hidden_size, self.intermediate_size)
        self.linear_fc2 = nn.Linear(self.intermediate_size, self.hidden_size)
        self.act_fn = ACT2FN[config.hidden_act]

    def __call__(self, hidden_state):
        return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_state)))


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        patch_volume = (
            self.in_channels
            * self.temporal_patch_size
            * self.patch_size
            * self.patch_size
        )
        self.proj = nn.Linear(patch_volume, self.embed_dim)

    def __call__(self, hidden_states):
        return self.proj(hidden_states.astype(self.proj.weight.dtype))


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.inv_freq = 1.0 / (theta ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))

    def __call__(self, seq_len: int):
        seq = mx.arange(seq_len).astype(self.inv_freq.dtype)
        return mx.outer(seq, self.inv_freq)


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, config, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_hidden_size = (
            self.hidden_size if use_postshuffle_norm else config.hidden_size
        )
        self.norm = nn.LayerNorm(norm_hidden_size, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, config.out_hidden_size)

    def __call__(self, x):
        if self.use_postshuffle_norm:
            x = x.reshape(-1, self.hidden_size)
            x = self.norm(x)
        else:
            x = self.norm(x)
            x = x.reshape(-1, self.hidden_size)
        x = self.linear_fc2(self.act_fn(self.linear_fc1(x)))
        return x


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3)
        self.proj = nn.Linear(self.dim, self.dim)

    def __call__(self, hidden_states, cu_seqlens, position_embeddings):
        seq_length = hidden_states.shape[0]
        qkv = self.qkv(hidden_states).reshape(
            seq_length, 3, self.num_heads, self.head_dim
        )
        query_states = qkv[:, 0]
        key_states = qkv[:, 1]
        value_states = qkv[:, 2]

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(
            query_states, key_states, cos, sin
        )

        lengths = np.diff(np.array(cu_seqlens)).tolist()
        attn_outputs = []
        start = 0
        for length in lengths:
            end = start + int(length)
            q = query_states[start:end].transpose(1, 0, 2)
            k = key_states[start:end].transpose(1, 0, 2)
            v = value_states[start:end].transpose(1, 0, 2)

            attn_weights = (
                q.astype(mx.float32) @ k.astype(mx.float32).transpose(0, 2, 1)
            ) * self.scaling
            attn_weights = mx.softmax(attn_weights, axis=-1).astype(q.dtype)
            attn_output = attn_weights @ v
            attn_outputs.append(attn_output.transpose(1, 0, 2))
            start = end

        attn_output = mx.concatenate(attn_outputs, axis=0)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(config=config)
        self.mlp = Qwen3VLVisionMLP(config=config)

    def __call__(self, hidden_states, cu_seqlens, position_embeddings):
        hidden_states = hidden_states + self.attn(
            self.norm1(hidden_states),
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )
        hidden_states = hidden_states + self.mlp(self.norm2(hidden_states))
        return hidden_states


class Qwen3VLVisionModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen3VLVisionPatchEmbed(config)
        self.pos_embed = nn.Embedding(
            config.num_position_embeddings, config.hidden_size
        )
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        head_dim = config.hidden_size // config.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)
        self.blocks = [Qwen3VLVisionBlock(config) for _ in range(config.depth)]
        self.merger = Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(config.deepstack_visual_indexes)
        self.deepstack_merger_list = [
            Qwen3VLVisionPatchMerger(config, use_postshuffle_norm=True)
            for _ in range(len(config.deepstack_visual_indexes))
        ]

    def rot_pos_emb(self, grid_thw):
        merge_size = self.spatial_merge_size
        grid_thw_list = np.array(grid_thw).astype(np.int64).tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = np.array(self.rotary_pos_emb(max_hw))

        all_embeddings = []
        for num_frames, height, width in grid_thw_list:
            merged_h = height // merge_size
            merged_w = width // merge_size

            block_rows = np.arange(merged_h)
            block_cols = np.arange(merged_w)
            intra_row = np.arange(merge_size)
            intra_col = np.arange(merge_size)

            row_idx = (
                block_rows[:, None, None, None] * merge_size
                + intra_row[None, None, :, None]
            )
            col_idx = (
                block_cols[None, :, None, None] * merge_size
                + intra_col[None, None, None, :]
            )

            row_idx = np.broadcast_to(
                row_idx, (merged_h, merged_w, merge_size, merge_size)
            ).reshape(-1)
            col_idx = np.broadcast_to(
                col_idx, (merged_h, merged_w, merge_size, merge_size)
            ).reshape(-1)

            coords = np.stack((row_idx, col_idx), axis=-1)
            if num_frames > 1:
                coords = np.repeat(coords, num_frames, axis=0)

            embeddings = freq_table[coords].reshape(coords.shape[0], -1)
            all_embeddings.append(embeddings)

        return mx.array(np.concatenate(all_embeddings, axis=0))

    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = np.array(grid_thw).astype(np.int64).tolist()
        weight = np.array(self.pos_embed.weight)
        merge_size = self.config.spatial_merge_size
        patch_pos_embeds = []

        for t, h, w in grid_thw_list:
            h_idxs = np.linspace(0, self.num_grid_per_side - 1, h, dtype=np.float32)
            w_idxs = np.linspace(0, self.num_grid_per_side - 1, w, dtype=np.float32)

            h_floor = np.floor(h_idxs).astype(np.int64)
            w_floor = np.floor(w_idxs).astype(np.int64)
            h_ceil = np.clip(h_floor + 1, 0, self.num_grid_per_side - 1)
            w_ceil = np.clip(w_floor + 1, 0, self.num_grid_per_side - 1)

            dh = h_idxs - h_floor
            dw = w_idxs - w_floor

            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side

            indices = [
                (base_h[:, None] + w_floor[None, :]).reshape(-1),
                (base_h[:, None] + w_ceil[None, :]).reshape(-1),
                (base_h_ceil[:, None] + w_floor[None, :]).reshape(-1),
                (base_h_ceil[:, None] + w_ceil[None, :]).reshape(-1),
            ]
            weights = [
                ((1 - dh)[:, None] * (1 - dw)[None, :]).reshape(-1),
                ((1 - dh)[:, None] * dw[None, :]).reshape(-1),
                (dh[:, None] * (1 - dw)[None, :]).reshape(-1),
                (dh[:, None] * dw[None, :]).reshape(-1),
            ]

            pos_embed = np.zeros((h * w, weight.shape[-1]), dtype=weight.dtype)
            for idx, idx_weight in zip(indices, weights):
                pos_embed += weight[idx] * idx_weight[:, None]

            pos_embed = np.repeat(pos_embed[None, :, :], t, axis=0)
            pos_embed = pos_embed.reshape(
                t,
                h // merge_size,
                merge_size,
                w // merge_size,
                merge_size,
                -1,
            )
            pos_embed = pos_embed.transpose(0, 1, 3, 2, 4, 5).reshape(
                -1, weight.shape[-1]
            )
            patch_pos_embeds.append(pos_embed)

        return mx.array(np.concatenate(patch_pos_embeds, axis=0))

    def __call__(self, hidden_states, grid_thw):
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.astype(hidden_states.dtype)

        rotary_pos_emb = self.rot_pos_emb(grid_thw)
        emb = mx.concatenate([rotary_pos_emb, rotary_pos_emb], axis=-1)
        position_embeddings = (
            mx.cos(emb).astype(hidden_states.dtype),
            mx.sin(emb).astype(hidden_states.dtype),
        )

        grid_np = np.array(grid_thw).astype(np.int64)
        repeated_lengths = np.repeat(grid_np[:, 1] * grid_np[:, 2], grid_np[:, 0])
        cu_seqlens = mx.array(
            np.pad(
                np.cumsum(repeated_lengths, dtype=np.int32), (1, 0), constant_values=0
            )
        )

        deepstack_features = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                merger_idx = self.deepstack_visual_indexes.index(layer_num)
                deepstack_feature = self.deepstack_merger_list[merger_idx](
                    hidden_states
                )
                deepstack_features.append(deepstack_feature)

        merged_hidden_states = self.merger(hidden_states)
        return BaseModelOutputWithDeepstackFeatures(
            last_hidden_state=hidden_states,
            pooler_output=merged_hidden_states,
            deepstack_features=deepstack_features,
        )


class Qwen3VLTextRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps: float = 1e-6):
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


class Qwen3VLTextRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        rope_parameters = getattr(config, "rope_parameters", None)
        if rope_parameters is None:
            rope_parameters = getattr(config, "rope_scaling", None)
        if rope_parameters is None:
            rope_parameters = {}

        self.rope_type = rope_parameters.get("rope_type", "default")
        self.mrope_section = rope_parameters.get("mrope_section", [24, 20, 20])
        base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2).astype(mx.float32) / dim))
        self.attention_scaling = 1.0

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_np = np.array(freqs)
        freqs_t = freqs_np[0].copy()
        for dim, offset in enumerate((1, 2), start=1):
            length = int(mrope_section[dim]) * 3
            freqs_t[..., offset:length:3] = freqs_np[dim, ..., offset:length:3]
        return mx.array(freqs_t)

    def __call__(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = mx.stack([position_ids, position_ids, position_ids], axis=0)

        inv_freq_expanded = self.inv_freq[None, None, :, None].astype(mx.float32)
        inv_freq_expanded = mx.broadcast_to(
            inv_freq_expanded,
            (3, position_ids.shape[1], inv_freq_expanded.shape[2], 1),
        )
        position_ids_expanded = position_ids[:, :, None, :].astype(mx.float32)
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(0, 1, 3, 2)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb) * self.attention_scaling
        sin = mx.sin(emb) * self.attention_scaling
        return cos.astype(x.dtype), sin.astype(x.dtype)


class Qwen3VLTextMLP(nn.Module):
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


class Qwen3VLTextAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout

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
        self.q_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3VLTextRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        output_attentions=False,
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


class Qwen3VLTextDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3VLTextMLP(config)
        self.input_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3VLTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def __call__(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        past_key_values=None,
        output_attentions=False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_outputs, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )

        hidden_states = residual + attn_outputs

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        outputs += (present_key_value,)
        return outputs


class Qwen3VLTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [
            Qwen3VLTextDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.norm = Qwen3VLTextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3VLTextRotaryEmbedding(config=config)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _update_causal_mask(self, attention_mask, input_tensor, past_seen_tokens):
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
                attention_mask_np = np.array(attention_mask)
                causal_mask_np = np.array(causal_mask)
                padding_mask = (causal_mask_np[..., :mask_length] == 0.0) * (
                    attention_mask_np[:, None, None, :] == 0.0
                )
                causal_mask_np[..., :mask_length] = np.ma.array(
                    data=causal_mask_np[..., :mask_length],
                    mask=padding_mask,
                ).filled(min_dtype)
                causal_mask = mx.array(causal_mask_np)
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

    def _deepstack_process(self, hidden_states, visual_pos_masks, visual_embeds):
        hidden_states_np = np.array(hidden_states)
        visual_mask_np = np.array(visual_pos_masks).astype(bool)
        visual_embeds_np = np.array(visual_embeds).astype(hidden_states_np.dtype)
        hidden_states_np[visual_mask_np] = (
            hidden_states_np[visual_mask_np] + visual_embeds_np
        )
        return mx.array(hidden_states_np)

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
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
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

        past_seen_tokens = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_seen_tokens = past_key_values.get_usable_length(inputs_embeds.shape[1])

        if position_ids is None:
            base_position_ids = mx.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1]
            )
            base_position_ids = mx.expand_dims(base_position_ids, axis=0)
            position_ids = mx.stack([base_position_ids] * 4, axis=0)
            position_ids = position_ids.reshape(4, 1, -1)
            position_ids = mx.broadcast_to(
                position_ids, (4, inputs_embeds.shape[0], inputs_embeds.shape[-2])
            )
        elif position_ids.ndim == 2:
            position_ids = mx.stack([position_ids] * 4, axis=0)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            rotary_position_ids = position_ids[1:]
        else:
            rotary_position_ids = position_ids

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, past_seen_tokens
        )
        position_embeddings = self.rotary_emb(inputs_embeds, rotary_position_ids)

        hidden_states = inputs_embeds
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if deepstack_visual_embeds is not None and layer_idx < len(
                deepstack_visual_embeds
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

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


class Qwen3VLModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)
        self.rope_deltas = None

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
    ):
        llm_grid_t = int(grid_thw[0]) // temp_merge_size
        llm_grid_h = int(grid_thw[1]) // spatial_merge_size
        llm_grid_w = int(grid_thw[2]) // spatial_merge_size

        image_seq_length = llm_grid_h * llm_grid_w * llm_grid_t
        position_width = np.arange(start_position, start_position + llm_grid_w).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = np.repeat(
            np.arange(start_position, start_position + llm_grid_h),
            llm_grid_w * llm_grid_t,
        )
        position_temporal = np.full(
            (image_seq_length,), start_position * time_interval, dtype=np.int64
        )
        vision_position_ids = np.stack(
            [position_temporal, position_height, position_width], axis=0
        )
        return mx.array(vision_position_ids)

    def get_rope_index(
        self,
        input_ids,
        mm_token_type_ids,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
    ):
        if video_grid_thw is not None:
            video_grid_np = np.repeat(
                np.array(video_grid_thw).astype(np.int64),
                np.array(video_grid_thw).astype(np.int64)[:, 0],
                axis=0,
            )
            video_grid_np[:, 0] = 1
            video_grid_thw = video_grid_np

        spatial_merge_size = self.config.vision_config.spatial_merge_size
        input_ids_np = np.array(input_ids).astype(np.int64)
        mm_token_type_ids_np = np.array(mm_token_type_ids).astype(np.int64)
        attention_mask_np = (
            None if attention_mask is None else np.array(attention_mask).astype(bool)
        )

        position_ids = np.zeros(
            (3, input_ids_np.shape[0], input_ids_np.shape[1]), dtype=np.int64
        )
        mrope_position_deltas = []
        grid_iters = {
            1: iter(np.array(image_grid_thw).astype(np.int64))
            if image_grid_thw is not None
            else None,
            2: iter(np.array(video_grid_thw).astype(np.int64))
            if video_grid_thw is not None
            else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids_np):
            input_token_type = mm_token_type_ids_np[batch_idx]
            if attention_mask_np is not None:
                current_input_ids = current_input_ids[attention_mask_np[batch_idx]]
                input_token_type = input_token_type[attention_mask_np[batch_idx]]

            input_type_group = []
            for key, group in itertools.groupby(
                enumerate(input_token_type.tolist()), lambda x: x[1]
            ):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        np.broadcast_to(
                            np.arange(current_pos, current_pos + text_len)[None, :],
                            (3, text_len),
                        )
                    )
                    current_pos += text_len
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = np.array(
                        self.get_vision_position_ids(
                            current_pos,
                            grid_thw,
                            1,
                            spatial_merge_size,
                        )
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += (
                        max(int(grid_thw[1]), int(grid_thw[2])) // spatial_merge_size
                    )

            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            if attention_mask_np is not None:
                position_ids[:, batch_idx, attention_mask_np[batch_idx]] = llm_positions
            else:
                position_ids[:, batch_idx] = llm_positions
            mrope_position_deltas.append(
                llm_positions.max() + 1 - len(current_input_ids)
            )

        rope_deltas = np.array(mrope_position_deltas, dtype=np.int64)[:, None]
        return mx.array(position_ids), mx.array(rope_deltas)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.get_image_features(pixel_values_videos, video_grid_thw)

    def get_image_features(self, pixel_values, image_grid_thw=None):
        vision_output = self.visual(pixel_values, grid_thw=image_grid_thw)
        image_embeds = vision_output.pooler_output
        split_sizes = (
            np.prod(np.array(image_grid_thw).astype(np.int64), axis=-1)
            // (self.visual.spatial_merge_size**2)
        ).tolist()

        image_embed_list = []
        start = 0
        for size in split_sizes:
            end = start + int(size)
            image_embed_list.append(image_embeds[start:end])
            start = end

        vision_output.pooler_output = image_embed_list
        return vision_output

    def get_placeholder_mask(
        self,
        input_ids,
        inputs_embeds,
        image_features=None,
        video_features=None,
    ):
        if input_ids is None:
            embeds_np = np.array(inputs_embeds)
            image_token_embed = np.array(
                self.get_input_embeddings()(
                    mx.array([self.config.image_token_id], dtype=mx.int32)
                )
            )[0]
            video_token_embed = np.array(
                self.get_input_embeddings()(
                    mx.array([self.config.video_token_id], dtype=mx.int32)
                )
            )[0]
            special_image_mask = np.all(embeds_np == image_token_embed, axis=-1)
            special_video_mask = np.all(embeds_np == video_token_embed, axis=-1)
        else:
            input_ids_np = np.array(input_ids)
            special_image_mask = input_ids_np == self.config.image_token_id
            special_video_mask = input_ids_np == self.config.video_token_id

        n_image_tokens = int(special_image_mask.sum())
        if image_features is not None and n_image_tokens != image_features.shape[0]:
            raise ValueError(
                "Image features and image tokens do not match, "
                f"tokens: {n_image_tokens}, features: {image_features.shape[0]}"
            )

        n_video_tokens = int(special_video_mask.sum())
        if video_features is not None and n_video_tokens != video_features.shape[0]:
            raise ValueError(
                "Video features and video tokens do not match, "
                f"tokens: {n_video_tokens}, features: {video_features.shape[0]}"
            )

        special_image_mask = mx.array(
            np.broadcast_to(special_image_mask[..., None], inputs_embeds.shape)
        )
        special_video_mask = mx.array(
            np.broadcast_to(special_video_mask[..., None], inputs_embeds.shape)
        )
        return special_image_mask, special_video_mask

    def compute_3d_position_ids(
        self,
        input_ids,
        inputs_embeds,
        image_grid_thw=None,
        video_grid_thw=None,
        attention_mask=None,
        past_key_values=None,
        mm_token_type_ids=None,
    ):
        past_key_values_length = (
            0 if past_key_values is None else past_key_values.get_seq_length()
        )
        has_multimodal = image_grid_thw is not None or video_grid_thw is not None
        if has_multimodal and mm_token_type_ids is None and input_ids is not None:
            raise ValueError(
                "Multimodal data was passed (via `image_grid_thw` or `video_grid_thw`) "
                "but `mm_token_type_ids` is missing."
            )

        can_compute_mrope = (
            input_ids is not None and mm_token_type_ids is not None and has_multimodal
        )

        if can_compute_mrope and (
            self.rope_deltas is None or past_key_values_length == 0
        ):
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
                mm_token_type_ids=mm_token_type_ids,
            )
            self.rope_deltas = rope_deltas
        elif self.rope_deltas is not None and (
            past_key_values_length > 0 or input_ids is None
        ):
            batch_size, seq_length, _ = inputs_embeds.shape
            if attention_mask is not None:
                position_ids = attention_mask.astype(mx.int32).cumsum(-1) - 1
                position_ids_np = np.array(position_ids)
                attention_mask_np = np.array(attention_mask)
                position_ids_np[attention_mask_np == 0] = 0
                position_ids = mx.array(position_ids_np)
                position_ids = mx.stack(
                    [position_ids, position_ids, position_ids], axis=0
                )
            else:
                base = mx.arange(
                    past_key_values_length, past_key_values_length + seq_length
                )
                position_ids = mx.expand_dims(base, axis=0)
                position_ids = mx.expand_dims(position_ids, axis=0)
                position_ids = mx.broadcast_to(
                    position_ids, (3, batch_size, seq_length)
                )

            rope_deltas = np.array(self.rope_deltas)
            if batch_size != rope_deltas.shape[0]:
                repeat_factor = batch_size // rope_deltas.shape[0]
                rope_deltas = np.repeat(rope_deltas, repeat_factor, axis=0)
            position_ids = position_ids + mx.array(rope_deltas.reshape(1, -1, 1))
        else:
            position_ids = None

        return position_ids

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
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

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = mx.concatenate(image_outputs.pooler_output, axis=0).astype(
                inputs_embeds.dtype
            )
            deepstack_image_embeds = image_outputs.deepstack_features
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds_np = np.array(inputs_embeds)
            image_mask_np = np.array(image_mask[..., 0]).astype(bool)
            inputs_embeds_np[image_mask_np] = np.array(image_embeds)
            inputs_embeds = mx.array(inputs_embeds_np)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw)
            video_embeds = mx.concatenate(video_outputs.pooler_output, axis=0).astype(
                inputs_embeds.dtype
            )
            deepstack_video_embeds = video_outputs.deepstack_features
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
            )
            inputs_embeds_np = np.array(inputs_embeds)
            video_mask_np = np.array(video_mask[..., 0]).astype(bool)
            inputs_embeds_np[video_mask_np] = np.array(video_embeds)
            inputs_embeds = mx.array(inputs_embeds_np)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask_np = np.array(image_mask[..., 0]).astype(bool)
            video_mask_np = np.array(video_mask[..., 0]).astype(bool)
            visual_pos_masks = mx.array(image_mask_np | video_mask_np)
            image_mask_joint = image_mask_np[image_mask_np | video_mask_np]
            video_mask_joint = video_mask_np[image_mask_np | video_mask_np]
            deepstack_visual_embeds = []
            for img_embed, vid_embed in zip(
                deepstack_image_embeds, deepstack_video_embeds
            ):
                img_embed_np = np.array(img_embed)
                vid_embed_np = np.array(vid_embed)
                joint = np.zeros(
                    (
                        int((image_mask_np | video_mask_np).sum()),
                        img_embed_np.shape[-1],
                    ),
                    dtype=img_embed_np.dtype,
                )
                joint[image_mask_joint] = img_embed_np
                joint[video_mask_joint] = vid_embed_np
                deepstack_visual_embeds.append(mx.array(joint))
        elif image_mask is not None:
            visual_pos_masks = mx.array(np.array(image_mask[..., 0]).astype(bool))
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = mx.array(np.array(video_mask[..., 0]).astype(bool))
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=DynamicCache.from_legacy_cache(past_key_values)
                if past_key_values is not None
                and not isinstance(past_key_values, Cache)
                else past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return Qwen3VLModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLForConditionalGeneration(nn.Module, MlxPretrainedMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = Qwen3VLModel(config)
        self.lm_head = nn.Linear(
            config.text_config.hidden_size,
            config.text_config.vocab_size,
            bias=False,
        )

    def _apply_pretrained_tensors(self, tensors: Dict[str, mx.array]) -> None:
        tensors = dict(tensors)
        patch_embed_weight_key = "model.visual.patch_embed.proj.weight"
        if (
            patch_embed_weight_key in tensors
            and tensors[patch_embed_weight_key].ndim == 5
        ):
            tensors[patch_embed_weight_key] = tensors[patch_embed_weight_key].reshape(
                tensors[patch_embed_weight_key].shape[0], -1
            )
        MlxPretrainedMixin._apply_pretrained_tensors(self, tensors)

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_image_features(self, pixel_values, image_grid_thw=None):
        return self.model.get_image_features(pixel_values, image_grid_thw)

    def get_video_features(self, pixel_values_videos, video_grid_thw=None):
        return self.model.get_video_features(pixel_values_videos, video_grid_thw)

    def __call__(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        logits_to_keep: int = 0,
    ):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
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
        logits = self.lm_head(logits_slice).astype(mx.float32)

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

        return Qwen3VLCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        mm_token_type_ids=None,
        is_first_iteration=False,
        **kwargs,
    ):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[..., -1:]

        model_inputs = {
            "input_ids": input_ids
            if inputs_embeds is None or past_key_values is not None
            else None,
            "inputs_embeds": inputs_embeds if past_key_values is None else None,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values
            if is_first_iteration or not use_cache
            else None,
            "pixel_values_videos": pixel_values_videos
            if is_first_iteration or not use_cache
            else None,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "mm_token_type_ids": mm_token_type_ids,
        }
        return model_inputs

    def generate(self, inputs: Dict, max_length: int, **kwargs):
        temp = kwargs.get("temp", 1.0)
        has_multimodal_inputs = (
            inputs.get("pixel_values") is not None
            or inputs.get("pixel_values_videos") is not None
        )

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
                if inputs.get("mm_token_type_ids") is not None:
                    inputs["mm_token_type_ids"] = mx.concatenate(
                        [
                            mx.array(inputs["mm_token_type_ids"]),
                            mx.zeros_like(next_token),
                        ],
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

            model_inputs = self.prepare_inputs_for_generation(
                input_ids=inputs["input_ids"],
                past_key_values=output.past_key_values,
                attention_mask=inputs["attention_mask"],
                inputs_embeds=None,
                position_ids=None,
                use_cache=use_cache,
                pixel_values=inputs.get("pixel_values"),
                pixel_values_videos=inputs.get("pixel_values_videos"),
                image_grid_thw=inputs.get("image_grid_thw"),
                video_grid_thw=inputs.get("video_grid_thw"),
                mm_token_type_ids=inputs.get("mm_token_type_ids"),
                is_first_iteration=False,
            )
            output = self(**model_inputs)

            next_token_logits = output.logits[:, -1, :]
            next_token = sample(next_token_logits)
            yield next_token
            generated_tokens += 1
