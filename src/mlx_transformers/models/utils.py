"""
Set of utility functions for the MLX Transformers models.
"""

# pylint: disable=c-extension-no-member
import math
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google
    BERT repo (identical to OpenAI GPT). Also see the Gaussian Error
    Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __call__(self, input: mx.array) -> mx.array:
        return (
            0.5
            * input
            * (
                1.0
                + mx.tanh(
                    math.sqrt(2.0 / math.pi) * (input + 0.044715 * mx.power(input, 3.0))
                )
            )
        )


class ReLUSquaredActivation(nn.Module):
    """
    Applies the relu^2 activation introduced in https://arxiv.org/abs/2109.08668v2
    """

    def __call__(self, input):
        relu_applied = nn.layers.activations.relu(input)
        squared = mx.square(relu_applied)
        return squared


ACT2FN = {
    "relu": nn.ReLU(),
    "relu2": ReLUSquaredActivation(),
    "gelu": nn.GELU(),
    "silu": nn.SiLU(),
    "gelu_new": NewGELUActivation(),
}


def get_extended_attention_mask(
    attention_mask: mx.array, input_shape: Tuple[int], dtype=None
):
    """
    Makes broadcastable attention and causal masks so that future
    and masked tokens are ignored.
    """

    if dtype is None:
        dtype = attention_mask.dtype

    if attention_mask.ndim == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.ndim == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or"
            "attention_mask (shape {attention_mask.shape})"
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


def convert(model_name: str, mlx_model: str, hgf_model_class) -> None:
    """
    Convert a Hugging Face model checkpoint to a MLX model.
    This simply loads the Hugging Face model and converts the tensors to numpy
    arrays and saves them in a .npz file.

    Args:
        model_name (str): Name of the Hugging Face model.
        mlx_model (str): mlx model class
        hgf_model_class (_type_): Hugging Face model class
    """
    model = hgf_model_class.from_pretrained(model_name)
    # save the tensors
    tensors = {key: tensor.numpy() for key, tensor in model.state_dict().items()}
    np.savez(mlx_model, **tensors)
