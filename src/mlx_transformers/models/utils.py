from typing import Tuple

import mlx.core as mx
import numpy as np


def get_extended_attention_mask(
    attention_mask: mx.array, input_shape: Tuple[int], dtype=None
):

    if dtype is None:
        dtype = attention_mask.dtype

    if attention_mask.ndim == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.ndim == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and the dtype's smallest value for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.

    extended_attention_mask = extended_attention_mask.astype(
        dtype=dtype
    )  # fp16 compatibility

    extended_attention_mask = (1.0 - extended_attention_mask) * np.finfo(np.float32).min
    return extended_attention_mask.astype(dtype=dtype)
