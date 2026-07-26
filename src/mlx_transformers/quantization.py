from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Mapping, Optional


QuantizationMode = Literal["affine", "mxfp4", "mxfp8", "nvfp4"]
QuantizationSource = Literal["checkpoint", "runtime"]

_MODE_DEFAULTS = {
    "affine": (64, 4),
    "mxfp4": (32, 4),
    "mxfp8": (32, 8),
    "nvfp4": (16, 4),
}
_AFFINE_GROUP_SIZES = {32, 64, 128}
_AFFINE_BITS = {2, 3, 4, 5, 6, 8}


@dataclass(frozen=True)
class QuantizationConfig:
    """Validated MLX weight-quantization settings.

    Defaults match ``mlx.nn.quantize``. The normalized values are explicit so
    callers can inspect and persist the exact runtime configuration.
    """

    group_size: Optional[int] = None
    bits: Optional[int] = None
    mode: QuantizationMode = "affine"
    quantize_input: bool = False

    def __post_init__(self) -> None:
        normalized_mode = str(self.mode).lower()
        if normalized_mode not in _MODE_DEFAULTS:
            supported = ", ".join(_MODE_DEFAULTS)
            raise ValueError(
                f"Unsupported quantization mode '{self.mode}'. "
                f"Choose one of: {supported}."
            )

        default_group_size, default_bits = _MODE_DEFAULTS[normalized_mode]
        group_size = default_group_size if self.group_size is None else self.group_size
        bits = default_bits if self.bits is None else self.bits

        if not isinstance(group_size, int) or isinstance(group_size, bool):
            raise ValueError("group_size must be an integer.")
        if not isinstance(bits, int) or isinstance(bits, bool):
            raise ValueError("bits must be an integer.")

        if normalized_mode == "affine":
            if group_size not in _AFFINE_GROUP_SIZES:
                raise ValueError(
                    "Affine quantization group_size must be 32, 64, or 128."
                )
            if bits not in _AFFINE_BITS:
                raise ValueError(
                    "Affine quantization bits must be one of 2, 3, 4, 5, 6, or 8."
                )
        else:
            expected_group_size, expected_bits = _MODE_DEFAULTS[normalized_mode]
            if (group_size, bits) != (expected_group_size, expected_bits):
                raise ValueError(
                    f"{normalized_mode} quantization requires "
                    f"group_size={expected_group_size} and bits={expected_bits}."
                )

        if self.quantize_input and normalized_mode not in {"mxfp8", "nvfp4"}:
            raise ValueError(
                "quantize_input=True is only supported for mode='nvfp4' or "
                "mode='mxfp8'."
            )

        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "group_size", group_size)
        object.__setattr__(self, "bits", bits)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "QuantizationConfig":
        if not isinstance(values, Mapping):
            raise ValueError("Checkpoint quantization metadata must be a mapping.")
        return cls(
            group_size=values.get("group_size"),
            bits=values.get("bits"),
            mode=values.get("mode", "affine"),
            quantize_input=values.get("quantize_input", False),
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QuantizationInfo:
    """The quantization configuration applied to a loaded model."""

    source: QuantizationSource
    group_size: int
    bits: int
    mode: QuantizationMode
    quantize_input: bool = False

    @classmethod
    def from_config(
        cls,
        config: QuantizationConfig,
        source: QuantizationSource,
    ) -> "QuantizationInfo":
        return cls(
            source=source,
            group_size=config.group_size,
            bits=config.bits,
            mode=config.mode,
            quantize_input=config.quantize_input,
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_quantization_info(model: Any) -> Optional[QuantizationInfo]:
    """Return normalized quantization metadata for a loaded model, if any."""

    info = getattr(model, "quantization_info", None)
    if info is not None and not isinstance(info, QuantizationInfo):
        raise TypeError("model.quantization_info is not a QuantizationInfo instance.")
    return info
