from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import mlx.core as mx
from transformers import AutoConfig, AutoTokenizer

from .models import (
    Gemma3ForCausalLM,
    LlamaForCausalLM,
    OpenELMForCausalLM,
    PersimmonForCausalLM,
    Phi3ForCausalLM,
    PhiForCausalLM,
    Qwen3ForCausalLM,
)
from .quantization import (
    QuantizationConfig,
    QuantizationInfo,
    get_quantization_info,
)


ARCHITECTURE_TO_CAUSAL_MODEL = {
    "Gemma3ForCausalLM": Gemma3ForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "OpenELMForCausalLM": OpenELMForCausalLM,
    "PersimmonForCausalLM": PersimmonForCausalLM,
    "Phi3ForCausalLM": Phi3ForCausalLM,
    "PhiForCausalLM": PhiForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}

MODEL_TYPE_TO_CAUSAL_MODEL = {
    "gemma3_text": Gemma3ForCausalLM,
    "llama": LlamaForCausalLM,
    "openelm": OpenELMForCausalLM,
    "persimmon": PersimmonForCausalLM,
    "phi": PhiForCausalLM,
    "phi3": Phi3ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}


@dataclass(frozen=True)
class LoadedCausalModel:
    model: Any
    tokenizer: Any
    quantization: Optional[QuantizationInfo]


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: List[int]


def resolve_causal_model_class(config: Any) -> Type[Any]:
    """Resolve an exported MLX causal model from a Transformers config."""

    architectures = getattr(config, "architectures", []) or []
    for architecture in architectures:
        model_class = ARCHITECTURE_TO_CAUSAL_MODEL.get(architecture)
        if model_class is not None:
            return model_class

    model_type = getattr(config, "model_type", None)
    model_class = MODEL_TYPE_TO_CAUSAL_MODEL.get(model_type)
    if model_class is not None:
        return model_class

    supported = ", ".join(sorted(MODEL_TYPE_TO_CAUSAL_MODEL))
    raise ValueError(
        "Unsupported causal model architecture. "
        f"architectures={architectures!r}, model_type={model_type!r}. "
        f"Supported model types: {supported}."
    )


def load_causal_model(
    model_name_or_path: str,
    *,
    tokenizer_name_or_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
    revision: str = "main",
    local_files_only: bool = False,
    token: Optional[str] = None,
    dtype: Optional[Any] = None,
    quantization: Optional[QuantizationConfig] = None,
) -> LoadedCausalModel:
    """Load a supported causal model and tokenizer for bounded inference.

    Pre-quantized MLX checkpoints are detected automatically. Pass
    ``quantization`` to quantize a regular safetensors checkpoint after load.
    """

    config = AutoConfig.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
        token=token,
        trust_remote_code=False,
    )
    model_class = resolve_causal_model_class(config)
    if (
        getattr(config, "model_type", None) == "openelm"
        and tokenizer_name_or_path is None
    ):
        raise ValueError(
            "OpenELM checkpoints do not bundle a tokenizer. Pass "
            "tokenizer_name_or_path explicitly."
        )

    model = model_class(config)
    model.from_pretrained(
        model_name_or_path,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
        token=token,
        dtype=dtype,
        quantization=quantization,
    )

    tokenizer_source = tokenizer_name_or_path or model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        cache_dir=cache_dir,
        revision=revision,
        local_files_only=local_files_only,
        token=token,
        trust_remote_code=False,
    )
    return LoadedCausalModel(
        model=model,
        tokenizer=tokenizer,
        quantization=get_quantization_info(model),
    )


def prepare_text_inputs(
    tokenizer: Any,
    prompt: str,
    *,
    use_chat_template: bool = True,
) -> Dict[str, mx.array]:
    """Tokenize one prompt for generation without creating Torch tensors."""

    if use_chat_template and getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="np",
        )
    else:
        encoded = tokenizer(prompt, return_tensors="np")
    return {key: mx.array(value) for key, value in encoded.items()}


def generate_text(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    use_chat_template: bool = True,
) -> GenerationResult:
    """Generate a bounded single-prompt response."""

    inputs = prepare_text_inputs(
        tokenizer,
        prompt,
        use_chat_template=use_chat_template,
    )
    token_ids = []
    for token in model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temp=temperature,
    ):
        mx.eval(token)
        if token.size != 1:
            raise ValueError("generate_text currently supports one prompt at a time.")
        token_ids.append(int(token.item()))

    text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return GenerationResult(text=text, token_ids=token_ids)
