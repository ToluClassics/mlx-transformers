import json
import os
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Union

from huggingface_hub import snapshot_download, HfFileSystem

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten

logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", None)

QuantizationPredicate = Optional[Callable[[str, Any], Union[bool, Dict[str, Any]]]]


class MlxPretrainedMixin:
    """Mixin class for loading pretrained models in MLX format."""

    @staticmethod
    def _load_checkpoint_config(download_path: Path) -> Dict[str, Any]:
        config_path = download_path / "config.json"
        if not config_path.exists():
            return {}

        with config_path.open("r", encoding="utf-8") as config_file:
            return json.load(config_file)

    @staticmethod
    def _get_checkpoint_quantization(
        checkpoint_config: Dict[str, Any], model_config: Any
    ) -> Optional[Dict[str, Any]]:
        checkpoint_quantization = checkpoint_config.get("quantization")
        if checkpoint_quantization is not None:
            return checkpoint_quantization

        checkpoint_quantization = checkpoint_config.get("quantization_config")
        if (
            isinstance(checkpoint_quantization, dict)
            and "group_size" in checkpoint_quantization
            and "bits" in checkpoint_quantization
        ):
            return checkpoint_quantization

        config_quantization = getattr(model_config, "quantization", None)
        if config_quantization is not None:
            return config_quantization

        config_quantization = getattr(model_config, "quantization_config", None)
        if (
            isinstance(config_quantization, dict)
            and "group_size" in config_quantization
            and "bits" in config_quantization
        ):
            return config_quantization

        return None

    @staticmethod
    def _is_prequantized_checkpoint(
        tensors: Dict[str, Any], checkpoint_quantization: Optional[Dict[str, Any]]
    ) -> bool:
        if checkpoint_quantization is None:
            return False
        return any(key.endswith(".scales") for key in tensors)

    @staticmethod
    def _quantize_model_for_checkpoint(
        model: "MlxPretrainedMixin",
        checkpoint_quantization: Dict[str, Any],
        tensor_keys: Set[str],
    ) -> None:
        def class_predicate(path: str, module: Any):
            if path in checkpoint_quantization:
                return checkpoint_quantization[path]
            if not hasattr(module, "to_quantized"):
                return False
            return f"{path}.scales" in tensor_keys

        nn.quantize(
            model,
            group_size=checkpoint_quantization["group_size"],
            bits=checkpoint_quantization["bits"],
            mode=checkpoint_quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    def _apply_pretrained_tensors(self, tensors: Dict[str, Any]) -> None:
        if hasattr(self, "load_weights"):
            self.load_weights(list(tensors.items()), strict=False)
        else:
            self.update(tree_unflatten(list(tensors.items())))

    def from_pretrained(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        revision: str = "main",
        float16: bool = False,
        trust_remote_code: bool = False,
        max_workers: int = 4,
        *,
        quantize: bool = False,
        group_size: Optional[int] = None,
        bits: Optional[int] = None,
        mode: str = "affine",
        quantize_input: bool = False,
        class_predicate: QuantizationPredicate = None,
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
            quantize: Whether to quantize the model after loading weights
            group_size: Quantization group size passed to ``mlx.nn.quantize``
            bits: Number of bits per quantized parameter
            mode: Quantization mode passed to ``mlx.nn.quantize``
            quantize_input: Whether to quantize supported layer inputs
            class_predicate: Optional predicate selecting which modules to quantize

        Returns:
            Self with loaded model weights
        """
        should_quantize = (
            quantize
            or group_size is not None
            or bits is not None
            or mode != "affine"
            or quantize_input
            or class_predicate is not None
        )
        if should_quantize and quantize_input and mode not in {"nvfp4", "mxfp8"}:
            raise ValueError(
                "quantize_input=True is only supported for mode='nvfp4' or "
                "mode='mxfp8'."
            )

        logger.info(
            f"Loading model from '{model_name_or_path}' "
            f"(revision={revision}, float16={float16}, quantize={should_quantize})"
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
                    allow_patterns=["*.safetensors", "config.json"],
                    cache_dir=cache_dir,
                    max_workers=max_workers,
                    revision=revision,
                    token=HF_TOKEN,
                )
            )

        if not safe_tensor_files:
            raise ValueError(f"No .safetensors files found at '{model_name_or_path}'")

        checkpoint_config = self._load_checkpoint_config(download_path)
        checkpoint_quantization = self._get_checkpoint_quantization(
            checkpoint_config, self.config
        )
        dtype = mx.float16 if float16 else mx.float32

        tensors = {}
        for file in safe_tensor_files:
            file_path = download_path / file
            tensors.update(mx.load(str(file_path)))

        prequantized_checkpoint = self._is_prequantized_checkpoint(
            tensors, checkpoint_quantization
        )
        if prequantized_checkpoint:
            if should_quantize:
                raise ValueError(
                    "Checkpoint already contains MLX quantized weights. "
                    "Load it without quantize/group_size/bits/mode arguments."
                )

            setattr(self.config, "quantization", checkpoint_quantization)
            self._quantize_model_for_checkpoint(
                self,
                checkpoint_quantization,
                set(tensors),
            )
        else:
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
                embed_weight_key = None
                for candidate in (
                    "model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight",
                ):
                    if candidate in tensors:
                        embed_weight_key = candidate
                        break

                if embed_weight_key is not None:
                    tensors["lm_head.weight"] = tensors[embed_weight_key]
                    embed_prefix = embed_weight_key.rsplit(".", 1)[0]
                    scales_key = f"{embed_prefix}.scales"
                    biases_key = f"{embed_prefix}.biases"
                    if scales_key in tensors:
                        tensors["lm_head.scales"] = tensors[scales_key]
                    if biases_key in tensors:
                        tensors["lm_head.biases"] = tensors[biases_key]

        self._apply_pretrained_tensors(tensors)
        if should_quantize and not prequantized_checkpoint:
            if quantize_input and class_predicate is None:
                linear_cls = nn.Linear

                def linear_only_predicate(_, module, linear_cls=linear_cls):
                    return isinstance(module, linear_cls)

                logger.info(
                    "quantize_input=True without class_predicate; restricting "
                    "quantization to mlx.nn.Linear layers."
                )
                class_predicate = linear_only_predicate

            logger.info(
                "Applying MLX quantization "
                "(group_size=%s, bits=%s, mode=%s, quantize_input=%s)",
                group_size,
                bits,
                mode,
                quantize_input,
            )
            nn.quantize(
                self,
                group_size=group_size,
                bits=bits,
                mode=mode,
                quantize_input=quantize_input,
                class_predicate=class_predicate,
            )
        self.eval()
        return self
