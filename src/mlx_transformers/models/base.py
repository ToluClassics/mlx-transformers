import importlib
import os
from typing import Callable, Optional

import mlx.core as mx
from huggingface_hub import HfFileSystem, hf_hub_download
from mlx.utils import tree_unflatten
from safetensors.numpy import load_file
from transformers import AutoConfig
from transformers.utils.import_utils import is_safetensors_available

CONFIG_FILE = "config.json"
WEIGHTS_FILE_NAME = "model.safetensors"


def _sanitize_keys(key):
    keys = key.split(".")
    return ".".join(keys[1:])


class MlxPretrainedMixin:

    def from_pretrained(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = "main",
        float16: bool = False,
        huggingface_model_architecture: Optional[Callable] = None,
    ):
        if huggingface_model_architecture:
            architecture = huggingface_model_architecture
        elif hasattr(self.config, "architectures"):
            architecture = self.config.architectures[0]
        else:
            raise ValueError("No architecture found for loading this model")

        transformers_module = importlib.import_module("transformers")
        _class = getattr(transformers_module, architecture, None)

        dtype = mx.float16 if float16 else mx.float32

        model = _class.from_pretrained(model_name_or_path)
        # # save the tensors
        tensors = {
            key: mx.array(tensor.numpy()).astype(dtype)
            for key, tensor in model.state_dict().items()
        }

        tensors = [(key, tensor) for key, tensor in tensors.items()]

        self.update(tree_unflatten(tensors))
