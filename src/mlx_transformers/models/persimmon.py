import mlx.core as mx
import mlx.nn as nn


class PersimmonRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        self.inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2) / dim))

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = mx.arange(self.max_seq_len_cached, dtype=mx.int64).astype(
            self.inv_freq.dtype
        )

        freqs = mx.outer(t, self.inv_freq)
        self.emb = mx.concatenate([freqs, freqs], axis=-1)

    def __call__(self, x, seq_len=None):
        cos = mx.cos(self.emb)
        sin = mx.sin(self.emb)

        return (cos[:seq_len], sin[:seq_len])
