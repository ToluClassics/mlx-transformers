import mlx.core as mx
import mlx.nn as nn


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = mx.ones((hidden_size,))
        self.variance_epsilon = eps

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(mx.float32)
        variance = hidden_states.power(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * mx.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.inv_freq = 1.0 / (
            self.base ** (mx.arange(0, self.dim, 2, dtype=mx.int64) / self.dim)
        )
        self.max_seq_len_cached = max_position_embeddings

        t = mx.arange(self.max_seq_len_cached).astype(mx.int64)
        t = t / self.scaling_factor
        freqs = mx.outer(t, inv_freq)

        emb = mx.concatenate([freqs, freqs], axis=-1)

    def __call__(self, x, position_ids):
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .astype(mx.float32)
            .broadcast_to(position_ids.shape[0], -1, 1)
        )
        position_ids_expanded = position_ids[:, None, :].astype(mx.float32)

        x_dtype = x.dtype

        freqs = inv_freq_expanded @ position_ids_expanded.transpose(0, 2, 1)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        cos = mx.cos(emb)
        sin = mx.sin(emb)

        return cos.astype(x_dtype), sin.astype(x_dtype)


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):

    def __call__(self, x, position_ids):
        position_ids = position_ids / self.scaling_factor
        cos, sin = super().__call__(x, position_ids)
        return cos, sin
