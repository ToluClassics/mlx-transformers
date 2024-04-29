import importlib
import logging
from typing import Callable, Optional

import mlx.core as mx
from mlx.utils import tree_unflatten

CONFIG_FILE = "config.json"
WEIGHTS_FILE_NAME = "model.safetensors"

logger = logging.getLogger(__name__)


class MlxPretrainedMixin:
    def from_pretrained(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        revision: Optional[str] = "main",
        float16: bool = False,
        huggingface_model_architecture: Optional[Callable] = None,
        trust_remote_code: bool = False,
        max_workers: int = 4,
    ):
        if huggingface_model_architecture:
            architecture = huggingface_model_architecture
        elif hasattr(self.config, "architectures"):
            architecture = self.config.architectures[0]
        else:
            raise ValueError("No architecture found for loading this model")

        transformers_module = importlib.import_module("transformers")
        _class = getattr(transformers_module, architecture, None)

        if not _class:
            raise ValueError(f"Could not find the class for {architecture}")

        dtype = mx.float16 if float16 else mx.float32

        logger.info(f"Loading model from {model_name_or_path}")
        model = _class.from_pretrained(
            model_name_or_path, trust_remote_code=trust_remote_code
        )

        # # save the tensors
        logger.info("Converting model tensors to Mx arrays")
        import concurrent.futures

        def convert_tensor(key, tensor, dtype):
            return key, mx.array(tensor.numpy()).astype(dtype)

        tensors = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for key, tensor in model.state_dict().items():
                future = executor.submit(convert_tensor, key, tensor, dtype)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                key, converted_tensor = future.result()
                tensors[key] = converted_tensor

        tensors = [(key, tensor) for key, tensor in tensors.items()]

        self.update(tree_unflatten(tensors))
