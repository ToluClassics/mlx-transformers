# Loading and running quantized models

MLX Transformers 0.3.0 supports two bounded quantized-inference workflows on
Apple silicon:

1. Load an MLX checkpoint that already contains quantized weights.
2. Load a regular safetensors checkpoint and quantize supported modules in
   memory before inference.

## Pre-quantized checkpoints

Use `load_causal_model` for a supported text-generation family. Quantization
metadata and `.scales` tensors are detected automatically:

```python
from mlx_transformers import generate_text, load_causal_model

loaded = load_causal_model(
    "mlx-community/Phi-3-mini-4k-instruct-4bit",
)

print(loaded.quantization)
result = generate_text(
    loaded.model,
    loaded.tokenizer,
    "Reply with one word: ready",
    max_new_tokens=8,
    temperature=0.0,
)
print(result.text)
```

For a local or cached checkpoint with networking disabled:

```python
loaded = load_causal_model(
    "/path/to/checkpoint",
    local_files_only=True,
)
```

The loader preserves the stored quantized tensor dtypes and exposes a
`QuantizationInfo` object whose `source` is `"checkpoint"`.

## Quantize after loading

Pass a validated `QuantizationConfig` for a regular checkpoint:

```python
import mlx.core as mx

from mlx_transformers import QuantizationConfig, load_causal_model

loaded = load_causal_model(
    "meta-llama/Llama-3.2-1B-Instruct",
    dtype=mx.float16,
    quantization=QuantizationConfig(
        group_size=64,
        bits=4,
        mode="affine",
    ),
)
```

The resulting `QuantizationInfo.source` is `"runtime"`. This path temporarily
holds the regular checkpoint in memory, so its peak memory can be materially
higher than direct pre-quantized loading. It changes the in-memory model only;
it does not write or publish a converted checkpoint.

The legacy `from_pretrained(..., quantize=True, group_size=..., bits=...)`
arguments remain supported for compatibility. Do not combine them with
`quantization=QuantizationConfig(...)`.

## Supported settings

Settings are validated before checkpoint loading:

| Mode | Group size | Bits | Quantized inputs |
| --- | --- | --- | --- |
| `affine` | 32, 64, or 128 | 2, 3, 4, 5, 6, or 8 | No |
| `mxfp4` | 32 | 4 | No |
| `mxfp8` | 32 | 8 | Optional |
| `nvfp4` | 16 | 4 | Optional |

The dimension quantized by MLX must be divisible by the chosen group size.
Runtime support also depends on the installed MLX version and Apple hardware.
The package bounds its supported MLX versions in `pyproject.toml`.

## Command-line generation

Pre-quantized checkpoint:

```bash
mlx-transformers-generate \
  --model mlx-community/Phi-3-mini-4k-instruct-4bit \
  --prompt "Reply with one word: ready" \
  --max-new-tokens 8
```

Runtime quantization:

```bash
mlx-transformers-generate \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --prompt "Reply with one word: ready" \
  --quantize --mode affine --group-size 64 --bits 4 \
  --max-new-tokens 8
```

The CLI requires both `--model` and `--prompt`, always uses a finite token
bound, never enables remote model code, and prints the effective quantization
metadata to standard error. Add `--local-files-only` to prohibit Hub network
resolution. Use `--raw-prompt` when a tokenizer chat template should not be
applied.

Supported causal model types are Gemma 3 text, Llama, OpenELM, Persimmon, Phi,
Phi-3, and Qwen3. OpenELM requires an explicit `--tokenizer` because its model
checkpoint does not bundle one. OpenELM and Persimmon are maintenance-only;
new verification work prioritizes active families listed in `SUPPORT.md`.
