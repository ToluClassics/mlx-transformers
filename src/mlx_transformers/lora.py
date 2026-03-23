import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten


@dataclass
class LoraConfig:
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.0
    scale: float = 10.0
    num_layers: Optional[int] = None
    target_modules: Optional[list[str]] = None


class LoRALinear(nn.Module):
    @staticmethod
    def from_linear(
        linear: nn.Module,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        scale: float = 10.0,
    ) -> "LoRALinear":
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims *= 32 // linear.bits

        lora_linear = LoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            alpha=alpha,
            dropout=dropout,
            scale=scale,
        )
        lora_linear.linear = linear
        lora_linear.linear.freeze()
        return lora_linear

    def to_linear(self, de_quantize: bool = False):
        linear = self.linear
        bias = "bias" in linear
        weight = linear.weight
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        dtype = weight.dtype
        if is_quantized:
            dtype = mx.float16
            weight = mx.dequantize(
                weight,
                linear.scales,
                linear.biases,
                linear.group_size,
                linear.bits,
            )

        output_dims, input_dims = weight.shape
        fused = nn.Linear(input_dims, output_dims, bias=bias)
        fused.weight = weight + (
            (self.scale * self.lora_b.T).astype(dtype) @ self.lora_a.T.astype(dtype)
        )
        if bias:
            fused.bias = linear.bias

        if is_quantized and not de_quantize:
            fused = nn.QuantizedLinear.from_linear(
                fused,
                linear.group_size,
                linear.bits,
            )

        return fused

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        scale: float = 10.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale * (alpha / r)

        init_scale = 1 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

    def __call__(self, x):
        y = self.linear(x)
        z = (self.dropout(x) @ self.lora_a) @ self.lora_b
        return y + (self.scale * z).astype(x.dtype)


def apply_lora(
    model: nn.Module,
    config: Optional[LoraConfig] = None,
) -> nn.Module:
    config = config or LoraConfig()
    target_modules = set(config.target_modules or _default_target_modules(model))
    layers = _get_transformer_layers(model)

    if config.num_layers is None:
        selected_layers = layers
    else:
        if config.num_layers < 1:
            raise ValueError("num_layers must be >= 1 when provided.")
        if config.num_layers > len(layers):
            raise ValueError(
                f"Requested {config.num_layers} LoRA layers but model only has {len(layers)}."
            )
        selected_layers = layers[-config.num_layers :]

    model.freeze()

    for layer in selected_layers:
        replacements = []
        for name, module in layer.named_modules():
            if not isinstance(module, (nn.Linear, nn.QuantizedLinear)):
                continue
            if not _matches_target(name, target_modules):
                continue
            replacements.append(
                (
                    name,
                    LoRALinear.from_linear(
                        module,
                        r=config.r,
                        alpha=config.alpha,
                        dropout=config.dropout,
                        scale=config.scale,
                    ),
                )
            )

        if replacements:
            layer.update_modules(tree_unflatten(replacements))

    return model


def save_lora_adapters(
    model: nn.Module,
    output_dir: str | Path,
    config: LoraConfig,
) -> Path:
    if not has_lora_layers(model):
        raise ValueError("Model does not contain any LoRA layers.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(output_dir / "adapters.safetensors"), adapter_weights)

    with (output_dir / "adapter_config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)

    return output_dir


def load_lora_adapters(
    model: nn.Module,
    adapter_path: str | Path,
) -> nn.Module:
    adapter_path = Path(adapter_path)
    config_path = adapter_path / "adapter_config.json"
    weights_path = adapter_path / "adapters.safetensors"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing adapter config: {config_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing adapter weights: {weights_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = LoraConfig(**json.load(handle))

    apply_lora(model, config)
    weights = mx.load(str(weights_path))
    if hasattr(model, "load_weights"):
        model.load_weights(list(weights.items()), strict=False)
    else:
        model.update(tree_unflatten(list(weights.items())))
    return model


def has_lora_layers(model: nn.Module) -> bool:
    return any(isinstance(module, LoRALinear) for _, module in model.named_modules())


def _matches_target(name: str, target_modules: set[str]) -> bool:
    return any(
        name == target or name.endswith(f".{target}") for target in target_modules
    )


def _get_transformer_layers(model: nn.Module) -> list[Any]:
    candidates = [model]
    for attr in ("model", "language_model", "decoder", "text_model"):
        module = getattr(model, attr, None)
        if module is not None:
            candidates.append(module)

    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if layers is not None:
            return list(layers)

    raise ValueError(
        "LoRA is currently supported only for models exposing a `.layers` stack."
    )


def _default_target_modules(model: nn.Module) -> list[str]:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)

    if model_type in {"llama", "qwen3", "phi", "phi3", "gemma3"}:
        return ["q_proj", "v_proj"]
    if model_type == "openelm":
        return ["qkv_proj"]

    raise ValueError(
        f"Unsupported model type for default LoRA targets: {model_type}. "
        "Pass `target_modules` explicitly."
    )
