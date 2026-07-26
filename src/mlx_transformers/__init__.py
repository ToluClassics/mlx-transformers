from importlib import import_module
from importlib.metadata import PackageNotFoundError, version

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

_LAZY_IMPORTS = {
    "GenerationResult": (".inference", "GenerationResult"),
    "LoadedCausalModel": (".inference", "LoadedCausalModel"),
    "QuantizationConfig": (".quantization", "QuantizationConfig"),
    "QuantizationInfo": (".quantization", "QuantizationInfo"),
    "generate_text": (".inference", "generate_text"),
    "get_quantization_info": (".quantization", "get_quantization_info"),
    "load_causal_model": (".inference", "load_causal_model"),
    "resolve_causal_model_class": (".inference", "resolve_causal_model_class"),
}


def __getattr__(name):
    try:
        module_name, attribute_name = _LAZY_IMPORTS[name]
    except KeyError as error:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from error

    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
