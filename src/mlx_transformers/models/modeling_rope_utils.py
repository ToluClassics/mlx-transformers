import math
from typing import Dict, Optional

from transformers import PretrainedConfig
import mlx.core as mx


def _get_rope_settings(config: PretrainedConfig) -> dict:
    rope_parameters = getattr(config, "rope_parameters", None)
    if rope_parameters is not None:
        return rope_parameters
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is not None:
        return rope_scaling
    return {}


def _compute_default_rope_parameters(
    config: PretrainedConfig, seq_len: Optional[int] = None, **rope_kwargs
) -> Dict[mx.array, float]:
    if config is not None and len(rope_kwargs) > 0:
        raise ValueError(
            "Unexpected arguments: `**rope_kwargs` and `config` are mutually exclusive in "
            f"`_compute_default_rope_parameters`, got `rope_kwargs`={rope_kwargs} and `config`={config}"
        )

    if len(rope_kwargs) > 0:
        base = rope_kwargs["base"]
        dim = rope_kwargs["dim"]
    elif config is not None:
        rope_settings = _get_rope_settings(config)
        base = rope_settings.get("rope_theta", getattr(config, "rope_theta", 10000.0))
        partial_rotary_factor = (
            config.partial_rotary_factor
            if hasattr(config, "partial_rotary_factor")
            else 1.0
        )
        head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)

    attention_factor = 1.0

    inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.int64) / dim))
    return inv_freq, attention_factor


def _compute_llama3_parameters(
    config: PretrainedConfig, seq_len: Optional[int] = None, **rope_kwargs
) -> Dict[mx.array, float]:
    inv_freq, attention_factor = _compute_default_rope_parameters(
        config, seq_len, **rope_kwargs
    )
    rope_settings = _get_rope_settings(config)
    factor = rope_settings["factor"]  # `8` in the original implementation
    low_freq_factor = rope_settings[
        "low_freq_factor"
    ]  # `1` in the original implementation
    high_freq_factor = rope_settings[
        "high_freq_factor"
    ]  # `4` in the original implementation
    old_context_len = rope_settings[
        "original_max_position_embeddings"
    ]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq

    inv_freq_llama = mx.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = mx.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama, attention_factor


ROPE_INIT_FUNCTIONS = {
    "default": _compute_default_rope_parameters,
    "llama3": _compute_llama3_parameters,
}
