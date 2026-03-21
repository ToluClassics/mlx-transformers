import os
import logging
from pathlib import Path
from typing import Optional

from huggingface_hub import snapshot_download, HfFileSystem

import mlx.core as mx
from mlx.utils import tree_unflatten, tree_flatten

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", None)


class MlxPretrainedMixin:
    """Mixin class for loading pretrained models in MLX format."""

    def from_pretrained(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        revision: str = "main",
        float16: bool = False,
        trust_remote_code: bool = False,
        max_workers: int = 4,
    ) -> "MlxPretrainedMixin":
        """
        Load a pretrained model from HuggingFace Hub or local path.

        Args:
            model_name_or_path: HuggingFace model name or path to local model directory
            cache_dir: Directory to store downloaded models
            revision: Git revision to use when downloading
            float16: Whether to convert model to float16
            trust_remote_code: Whether to trust remote code when loading
            max_workers: Number of worker threads for tensor conversion

        Returns:
            Self with loaded model weights
        """
        logger.info(
            f"Loading model from '{model_name_or_path}' "
            f"(revision={revision}, float16={float16})"
        )

        local_path = Path(model_name_or_path)
        if local_path.is_dir():
            download_path = local_path
            safe_tensor_files = [f.name for f in local_path.glob("*.safetensors")]
        else:
            fs = HfFileSystem()
            remote_files = fs.glob(
                f"{model_name_or_path}/*.safetensors",
                revision=revision,
            )
            safe_tensor_files = [f.split("/")[-1] for f in remote_files]

            if not safe_tensor_files:
                raise ValueError(
                    f"No .safetensors files found for model '{model_name_or_path}'"
                )

            download_path = Path(
                snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns="*.safetensors",
                    cache_dir=cache_dir,
                    max_workers=max_workers,
                    revision=revision,
                    token=HF_TOKEN,
                )
            )

        if not safe_tensor_files:
            raise ValueError(f"No .safetensors files found at '{model_name_or_path}'")

        dtype = mx.float16 if float16 else mx.float32

        tensors = {}
        for file in safe_tensor_files:
            file_path = download_path / file
            tensors.update(mx.load(str(file_path)))

        tensors = {k: v.astype(dtype) for k, v in tensors.items()}

        model_param_keys = set(param[0] for param in tree_flatten(self.parameters()))
        ignored_tensor_keys = sorted(set(tensors) - model_param_keys)
        if ignored_tensor_keys:
            logger.info(
                "Ignoring %d pretrained tensors that do not map to MLX parameters: %s",
                len(ignored_tensor_keys),
                ", ".join(ignored_tensor_keys),
            )
            tensors = {k: v for k, v in tensors.items() if k in model_param_keys}

        if self.config.tie_word_embeddings:
            missing_keys = model_param_keys - set(tensors.keys())

            # Architecture-specific tied-embedding resolution.
            # Add branches here for other architectures as needed.
            if "lm_head.weight" in missing_keys:
                tensors["lm_head.weight"] = tensors["model.embed_tokens.weight"]

        self.update(tree_unflatten(list(tensors.items())))
        self.eval()
        return self
