import json
import logging
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union

from huggingface_hub import snapshot_download
from huggingface_hub.errors import (
    GatedRepoError,
    HfHubHTTPError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten, tree_flatten

logger = logging.getLogger(__name__)

QuantizationPredicate = Optional[Callable[[str, Any], Union[bool, Dict[str, Any]]]]


class MlxPretrainedMixin:
    """Mixin class for loading pretrained models in MLX format."""

    _CHECKPOINT_PATTERNS = [
        "*.safetensors",
        "**/*.safetensors",
        "*.safetensors.index.json",
        "config.json",
    ]
    _IGNORED_CHECKPOINT_SUFFIXES = (
        "embeddings.position_ids",
        "embeddings.token_type_ids",
        "pooler.dense.bias",
        "pooler.dense.weight",
        "rotary_emb.inv_freq",
    )
    _DERIVED_MODEL_PARAMETER_SUFFIXES = (
        "embeddings.position_ids",
        "embeddings.token_type_ids",
        "embed_tokens.embed_scale",
        "rotary_emb.cos",
        "rotary_emb.emb",
        "rotary_emb.inv_freq",
        "rotary_emb.inv_freq.full_attention",
        "rotary_emb.inv_freq.sliding_attention",
        "rotary_emb.original_inv_freq",
        "rotary_emb.sin",
        "rotary_pos_emb.inv_freq",
    )

    def _generate_tokens(
        self,
        inputs: Dict[str, Any],
        *,
        max_new_tokens: Optional[int] = None,
        max_length: Optional[int] = None,
        temp: float = 1.0,
        use_cache: bool = True,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        pad_token_id: Optional[int] = None,
        persistent_input_keys=(),
        sequence_input_fill_values=None,
    ):
        """Generate a finite stream of next-token arrays for text models."""
        if max_new_tokens is not None and max_length is not None:
            raise ValueError("Pass only one of max_new_tokens or max_length.")
        if max_new_tokens is None:
            if max_length is None:
                raise ValueError("max_new_tokens is required.")
            warnings.warn(
                "max_length currently means generated-token count and is "
                "deprecated; use max_new_tokens instead.",
                DeprecationWarning,
                stacklevel=3,
            )
            max_new_tokens = max_length
        if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a non-negative integer.")
        if temp < 0:
            raise ValueError("temp must be non-negative.")
        if max_new_tokens == 0:
            return

        model_inputs = dict(inputs)
        sequence_input_fill_values = sequence_input_fill_values or {}
        persistent_inputs = {
            key: model_inputs.get(key)
            for key in persistent_input_keys
            if model_inputs.get(key) is not None
        }
        sequence_inputs = {
            key: mx.array(model_inputs[key])
            for key in sequence_input_fill_values
            if model_inputs.get(key) is not None
        }
        if "input_ids" not in model_inputs:
            raise ValueError("generate requires input_ids.")

        input_ids = model_inputs["input_ids"]
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence].")

        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is None:
            attention_mask = mx.ones_like(input_ids)
        else:
            attention_mask = mx.array(attention_mask)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids.")

        if eos_token_id is None:
            eos_token_id = getattr(self.config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_ids = ()
        elif isinstance(eos_token_id, int):
            eos_token_ids = (eos_token_id,)
        else:
            eos_token_ids = tuple(eos_token_id)

        if pad_token_id is None:
            pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_ids[0] if eos_token_ids else 0

        def sample(logits):
            if temp == 0:
                return mx.argmax(logits, axis=-1)
            return mx.random.categorical(logits * (1 / temp))

        def last_token_logits(logits, mask):
            positions = mx.arange(mask.shape[-1], dtype=mx.int32)
            last_indices = mx.argmax(mask.astype(mx.int32) * positions, axis=-1)
            return logits[mx.arange(logits.shape[0]), last_indices]

        model_inputs["attention_mask"] = attention_mask
        model_inputs["use_cache"] = use_cache
        output = self(**model_inputs)
        logits = last_token_logits(output.logits, attention_mask)
        finished = mx.zeros((input_ids.shape[0],), dtype=mx.bool_)
        full_input_ids = input_ids

        for generated_index in range(max_new_tokens):
            sampled_token = sample(logits)
            next_token = mx.where(finished, pad_token_id, sampled_token)
            yield next_token

            is_eos = mx.zeros_like(finished)
            for token_id in eos_token_ids:
                is_eos = mx.logical_or(is_eos, next_token == token_id)
            new_finished = mx.logical_or(finished, is_eos)
            if bool(mx.all(new_finished).item()):
                return
            if generated_index + 1 == max_new_tokens:
                return

            full_input_ids = mx.concatenate(
                [full_input_ids, mx.expand_dims(next_token, axis=-1)],
                axis=-1,
            )
            next_attention = mx.logical_not(finished).astype(attention_mask.dtype)
            attention_mask = mx.concatenate(
                [attention_mask, mx.expand_dims(next_attention, axis=-1)],
                axis=-1,
            )
            for key, value in sequence_inputs.items():
                fill_value = sequence_input_fill_values[key]
                extension = mx.full(
                    (value.shape[0], 1),
                    fill_value,
                    dtype=value.dtype,
                )
                sequence_inputs[key] = mx.concatenate(
                    [value, extension],
                    axis=-1,
                )
            finished = new_finished

            past_key_values = output.past_key_values if use_cache else None
            prepare_kwargs = {
                **persistent_inputs,
                **sequence_inputs,
            }
            if persistent_inputs:
                prepare_kwargs["is_first_iteration"] = False
            model_inputs = self.prepare_inputs_for_generation(
                input_ids=full_input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=None,
                use_cache=use_cache,
                **prepare_kwargs,
            )
            output = self(**model_inputs)
            logits = output.logits[:, -1, :]

    def _normalize_pretrained_tensors(
        self, tensors: Dict[str, mx.array]
    ) -> Dict[str, mx.array]:
        """Map upstream checkpoint names to this model's parameter topology."""
        return tensors

    @staticmethod
    def _looks_like_local_path(model_name_or_path: str) -> bool:
        return Path(model_name_or_path).expanduser().is_absolute() or (
            model_name_or_path.startswith(("./", "../", "~"))
        )

    @classmethod
    def _resolve_checkpoint_path(
        cls,
        model_name_or_path: str,
        *,
        cache_dir: Optional[str],
        revision: str,
        local_files_only: bool,
        token: Optional[str],
        max_workers: int,
    ) -> Path:
        local_path = Path(model_name_or_path).expanduser()
        if local_path.is_dir():
            return local_path
        if cls._looks_like_local_path(model_name_or_path):
            raise FileNotFoundError(
                f"Local model directory does not exist: '{model_name_or_path}'"
            )

        try:
            return Path(
                snapshot_download(
                    repo_id=model_name_or_path,
                    allow_patterns=cls._CHECKPOINT_PATTERNS,
                    cache_dir=cache_dir,
                    local_files_only=local_files_only,
                    max_workers=max_workers,
                    revision=revision,
                    token=token,
                )
            )
        except GatedRepoError as error:
            raise PermissionError(
                f"Model '{model_name_or_path}' is gated or private. "
                "Pass token=... with access to the repository."
            ) from error
        except RevisionNotFoundError as error:
            raise ValueError(
                f"Revision '{revision}' was not found for model "
                f"'{model_name_or_path}'."
            ) from error
        except LocalEntryNotFoundError as error:
            raise FileNotFoundError(
                f"No cached snapshot for model '{model_name_or_path}' at "
                f"revision '{revision}'. Disable local_files_only or download "
                "the checkpoint explicitly."
            ) from error
        except RepositoryNotFoundError as error:
            raise FileNotFoundError(
                f"Model repository '{model_name_or_path}' was not found."
            ) from error
        except HfHubHTTPError as error:
            raise RuntimeError(
                f"Unable to resolve model '{model_name_or_path}' from the "
                "Hugging Face Hub."
            ) from error

    @staticmethod
    def _validate_relative_checkpoint_path(relative_path: str) -> Path:
        path = Path(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Invalid safetensors shard path in checkpoint index: "
                f"'{relative_path}'"
            )
        return path

    @classmethod
    def _discover_safetensor_files(cls, checkpoint_path: Path) -> List[Path]:
        index_files = sorted(checkpoint_path.rglob("*.safetensors.index.json"))
        if len(index_files) > 1:
            raise ValueError(
                f"Multiple safetensors index files found in '{checkpoint_path}'."
            )

        if index_files:
            try:
                index_data = json.loads(index_files[0].read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Unable to read safetensors index '{index_files[0]}'."
                ) from error

            weight_map = index_data.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise ValueError(
                    f"Safetensors index '{index_files[0]}' has no weight_map."
                )

            relative_files = sorted(
                {
                    cls._validate_relative_checkpoint_path(file_name)
                    for file_name in weight_map.values()
                    if isinstance(file_name, str)
                },
                key=str,
            )
            if not relative_files:
                raise ValueError(
                    f"Safetensors index '{index_files[0]}' has no valid shards."
                )

            shard_files = [checkpoint_path / file for file in relative_files]
            missing_shards = [file for file in shard_files if not file.is_file()]
            if missing_shards:
                missing = ", ".join(str(file) for file in missing_shards)
                raise FileNotFoundError(
                    f"Safetensors index references missing shard(s): {missing}"
                )
            return shard_files

        safe_tensor_files = sorted(checkpoint_path.rglob("*.safetensors"))
        if safe_tensor_files:
            return safe_tensor_files

        if any(checkpoint_path.rglob("*.bin")):
            raise ValueError(
                f"Checkpoint '{checkpoint_path}' contains PyTorch .bin weights, "
                "but mlx-transformers supports safetensors only."
            )
        raise ValueError(f"No .safetensors files found at '{checkpoint_path}'")

    @staticmethod
    def _format_keys(keys: Set[str], limit: int = 10) -> str:
        sorted_keys = sorted(keys)
        result = ", ".join(sorted_keys[:limit])
        if len(sorted_keys) > limit:
            result += f", ... ({len(sorted_keys) - limit} more)"
        return result

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
            self.load_weights(list(tensors.items()), strict=True)
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
        local_files_only: bool = False,
        token: Optional[str] = None,
        dtype: Optional[Any] = None,
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
            local_files_only: Resolve Hub model IDs from the local cache only
            token: Explicit Hugging Face token for a gated/private repository
            dtype: Explicit MLX floating-point dtype for loaded tensors
            quantize: Whether to quantize the model after loading weights
            group_size: Quantization group size passed to ``mlx.nn.quantize``
            bits: Number of bits per quantized parameter
            mode: Quantization mode passed to ``mlx.nn.quantize``
            quantize_input: Whether to quantize supported layer inputs
            class_predicate: Optional predicate selecting which modules to quantize

        Returns:
            Self with loaded model weights
        """
        if trust_remote_code:
            warnings.warn(
                "trust_remote_code has no effect in mlx-transformers; the "
                "loader never executes code from a model repository.",
                UserWarning,
                stacklevel=2,
            )
        if float16 and dtype is not None:
            raise ValueError("Pass only one of float16=True or dtype=....")
        if dtype is not None and dtype not in {
            mx.float16,
            mx.bfloat16,
            mx.float32,
        }:
            raise ValueError(
                "dtype must be one of mx.float16, mx.bfloat16, or mx.float32."
            )

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

        download_path = self._resolve_checkpoint_path(
            model_name_or_path,
            cache_dir=cache_dir,
            revision=revision,
            local_files_only=local_files_only,
            token=token,
            max_workers=max_workers,
        )
        safe_tensor_files = self._discover_safetensor_files(download_path)

        checkpoint_config = self._load_checkpoint_config(download_path)
        checkpoint_quantization = self._get_checkpoint_quantization(
            checkpoint_config, self.config
        )
        load_dtype = (
            dtype if dtype is not None else (mx.float16 if float16 else mx.float32)
        )

        tensors = {}
        for file in safe_tensor_files:
            shard_tensors = mx.load(str(file))
            duplicate_keys = set(tensors).intersection(shard_tensors)
            if duplicate_keys:
                raise ValueError(
                    "Duplicate tensor key(s) found across safetensors shards: "
                    f"{self._format_keys(duplicate_keys)}"
                )
            tensors.update(shard_tensors)

        tensors = self._normalize_pretrained_tensors(tensors)
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
            tensors = {k: v.astype(load_dtype) for k, v in tensors.items()}

        model_params = dict(tree_flatten(self.parameters()))
        model_param_keys = set(model_params)
        unexpected_tensor_keys = set(tensors) - model_param_keys
        ignored_tensor_keys = {
            key
            for key in unexpected_tensor_keys
            if key.endswith(self._IGNORED_CHECKPOINT_SUFFIXES)
        }
        if ignored_tensor_keys:
            logger.info(
                "Ignoring %d pretrained tensors that do not map to MLX parameters: %s",
                len(ignored_tensor_keys),
                self._format_keys(ignored_tensor_keys),
            )
            tensors = {
                key: value
                for key, value in tensors.items()
                if key not in ignored_tensor_keys
            }

        unsupported_tensor_keys = unexpected_tensor_keys - ignored_tensor_keys
        if unsupported_tensor_keys:
            raise ValueError(
                "Checkpoint contains tensor key(s) that do not map to this "
                f"model: {self._format_keys(unsupported_tensor_keys)}"
            )

        if getattr(self.config, "tie_word_embeddings", False):
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

            if "lm_head.weight" in tensors:
                for candidate in (
                    "model.shared.weight",
                    "model.encoder.embed_tokens.weight",
                    "model.decoder.embed_tokens.weight",
                    "model.embed_tokens.weight",
                    "model.language_model.embed_tokens.weight",
                ):
                    if candidate in missing_keys:
                        tensors[candidate] = tensors["lm_head.weight"]

        missing_keys = model_param_keys - set(tensors)
        derived_parameter_keys = {
            key
            for key in missing_keys
            if key.endswith(self._DERIVED_MODEL_PARAMETER_SUFFIXES)
        }
        if derived_parameter_keys:
            logger.info(
                "Keeping %d deterministic model parameters derived from config: %s",
                len(derived_parameter_keys),
                self._format_keys(derived_parameter_keys),
            )
            tensors.update({key: model_params[key] for key in derived_parameter_keys})

        missing_keys = model_param_keys - set(tensors)
        if missing_keys:
            raise ValueError(
                "Checkpoint is missing required model tensor key(s): "
                f"{self._format_keys(missing_keys)}"
            )

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
