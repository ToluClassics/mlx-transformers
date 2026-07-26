from importlib.metadata import PackageNotFoundError, version

from .inference import (
    GenerationResult,
    LoadedCausalModel,
    generate_text,
    load_causal_model,
    resolve_causal_model_class,
)
from .quantization import (
    QuantizationConfig,
    QuantizationInfo,
    get_quantization_info,
)

try:
    __version__ = version("mlx-transformers")
except PackageNotFoundError:
    __version__ = "0+unknown"

__all__ = [
    "GenerationResult",
    "LoadedCausalModel",
    "QuantizationConfig",
    "QuantizationInfo",
    "__version__",
    "generate_text",
    "get_quantization_info",
    "load_causal_model",
    "resolve_causal_model_class",
]
