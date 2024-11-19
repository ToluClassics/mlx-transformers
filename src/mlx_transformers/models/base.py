import os
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Dict, Any
from huggingface_hub import snapshot_download
from huggingface_hub import HfFileSystem

import mlx.core as mx
from mlx.utils import tree_unflatten

logger = logging.getLogger(__name__)
fs = HfFileSystem()

HF_TOKEN = os.getenv("HF_TOKEN", None)

@dataclass
class ModelLoadingConfig:
    """Configuration for model loading parameters."""
    model_name_or_path: str
    cache_dir: Optional[str] = None
    revision: str = "main"
    float16: bool = False
    trust_remote_code: bool = False
    max_workers: int = 4

class MlxPretrainedMixin:
    """Mixin class for loading pretrained models in MLX format."""

    def from_pretrained(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        revision: str = "main",
        float16: bool = False,
        huggingface_model_architecture: Optional[str] = None,
        trust_remote_code: bool = False,
        max_workers: int = 4,
    ) -> "MlxPretrainedMixin":
        """
        Load a pretrained model from HuggingFace Hub or local path.
        
        Args:
            model_name_or_path: HuggingFace model name or path to local model
            cache_dir: Directory to store downloaded models
            revision: Git revision to use when downloading
            float16: Whether to convert model to float16
            huggingface_model_architecture: Custom model architecture class
            trust_remote_code: Whether to trust remote code when loading
            max_workers: Number of worker threads for tensor conversion
            
        Returns:
            Self with loaded model weights
        """
        config = ModelLoadingConfig(
            model_name_or_path=model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            float16=float16,
            trust_remote_code=trust_remote_code,
            max_workers=max_workers
        )
        
        logger.info(f"Loading model from {config.model_name_or_path}")

        safe_tensor_files = fs.glob(f"{config.model_name_or_path}/*.safetensors")
        safe_tensor_files = [f.split("/")[-1] for f in safe_tensor_files]

        if not safe_tensor_files:
            raise ValueError("No safe tensor files found for this model")
        
        download_path = snapshot_download(repo_id=config.model_name_or_path, 
                                          allow_patterns="*.safetensors",
                                          max_workers=config.max_workers,
                                          token=HF_TOKEN)
        dtype = mx.float16 if config.float16 else mx.float32

        tensors = {}
        for file in safe_tensor_files:
            file_path = Path(download_path) / file
            with file_path.open("rb") as f:
                state_dict = mx.load(f)

            tensors.update(state_dict)
        
        tensors = {k: v.astype(dtype) for k, v in tensors.items()}

        # Update model weights
        self.update(tree_unflatten(list(tensors.items())))
        return self