# MLX Transformers

[![PyPI](https://img.shields.io/pypi/v/mlx-transformers?color=red)](https://pypi.org/project/mlx-transformers/)

MLX implementations of Hugging Face-style models for Apple Silicon.

## Installation

```bash
pip install mlx-transformers
```

Install only the optional features you use:

```bash
pip install "mlx-transformers[tokenizers]"
pip install "mlx-transformers[vision]"
pip install "mlx-transformers[chat]"
```

For local development:

```bash
python3.12 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[test,examples,chat]"
```

MLX requires Apple silicon and macOS. Verify that the environment can execute
on Metal before running model tests:

```bash
python -c 'import mlx.core as mx; x = mx.array([1, 2, 3]); print(mx.sum(x).item())'
```

## Quick Start

```python
import mlx.core as mx
from transformers import AutoConfig, AutoTokenizer

from mlx_transformers.models import BertModel

model_name = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
config = AutoConfig.from_pretrained(model_name)

model = BertModel(config)
model.from_pretrained(model_name)

inputs = tokenizer("Hello from MLX", return_tensors="np")
inputs = {k: mx.array(v) for k, v in inputs.items()}

outputs = model(**inputs)
```

## Quantized Inference

MLX Transformers can auto-detect an MLX pre-quantized checkpoint and run it
without extra loader flags:

```python
from mlx_transformers import generate_text, load_causal_model

loaded = load_causal_model(
    "mlx-community/Phi-3-mini-4k-instruct-4bit",
)
result = generate_text(
    loaded.model,
    loaded.tokenizer,
    "Explain weight quantization in one sentence.",
    max_new_tokens=64,
)
print(loaded.quantization)
print(result.text)
```

To quantize a regular safetensors checkpoint in memory after loading:

```python
import mlx.core as mx

from mlx_transformers import QuantizationConfig, load_causal_model

loaded = load_causal_model(
    model_name,
    dtype=mx.float16,
    quantization=QuantizationConfig(
        group_size=64,
        bits=4,
        mode="affine",
    ),
)
```

The existing model-specific loader flags remain supported:

```python
model.from_pretrained(model_name, quantize=True, group_size=64, bits=4)
```

The installed CLI offers the same two paths:

```bash
# Auto-detect a pre-quantized MLX checkpoint.
mlx-transformers-generate \
  --model mlx-community/Phi-3-mini-4k-instruct-4bit \
  --prompt "Explain attention masking." \
  --max-new-tokens 64

# Quantize a regular checkpoint after loading it.
mlx-transformers-generate \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --prompt "Explain attention masking." \
  --quantize --group-size 64 --bits 4 \
  --max-new-tokens 64
```

On-load quantization temporarily materializes the regular checkpoint before
replacing supported layers, so peak memory is higher than loading an already
quantized checkpoint. Prefer a reviewed pre-quantized MLX checkpoint for large
models. See [docs/load_model.md](docs/load_model.md) for supported modes,
offline use, metadata inspection, and safety constraints.

Generation is finite and uses Hugging Face-style `max_new_tokens`:

```python
for token_ids in model.generate(
    inputs["input_ids"],
    attention_mask=inputs.get("attention_mask"),
    max_new_tokens=64,
    temp=0.0,
):
    print(token_ids)
```

The generator supports batched left- or right-padded prompts, per-sequence
end-of-sequence handling, and cached or uncached decoding. The legacy
`max_length` argument remains as a deprecated generated-token-count alias.

For large multimodal Gemma 3 checkpoints, prefer `dtype=mx.bfloat16` on
supported Apple silicon:

```python
model.from_pretrained(model_name, dtype=mx.bfloat16)
```

Offline or authenticated loading:

```python
# Resolve an already-cached Hub snapshot without network access.
model.from_pretrained(model_name, local_files_only=True)

# Local checkpoint directories are also supported.
model.from_pretrained("/path/to/local/checkpoint")

# Pass credentials explicitly for a reviewed gated/private repository.
model.from_pretrained("org/private-model", token=token)
```

The loader supports safetensors checkpoints and shard indexes. It rejects
missing required weights, duplicate shard keys, incompatible extra weights,
and PyTorch `.bin`-only checkpoints instead of leaving model parameters
silently initialized. `trust_remote_code` is not used: MLX Transformers never
executes code from a model repository.

## Model Support

Real-checkpoint verification currently covers BERT, Llama, Phi-3, Qwen3,
Gemma 3, and M2M100/NLLB paths. Phi, Qwen3-VL, RoBERTa, XLM-RoBERTa, OpenELM,
Persimmon, and Fuyu remain experimental because at least one important
real-checkpoint path is still unverified.

See [SUPPORT.md](SUPPORT.md) for exact checkpoints/tasks, compatibility bounds,
dtype limitations, generation semantics, and the verified/experimental
promotion policy. OpenELM, Persimmon, and Fuyu are maintenance-only: existing
behavior retains bounded regression coverage, but new compatibility and
feature work prioritizes the active model families.

## Examples

Phi-3:

```bash
python examples/text_generation/phi3_generation.py \
  --model-name microsoft/Phi-3-mini-4k-instruct \
  --prompt "Explain attention masking." \
  --max-tokens 128 \
  --temp 0.0
```

Qwen3-VL:

```bash
python examples/text_generation/qwen3_vl_generation.py \
  --model-name Qwen/Qwen3-VL-2B-Instruct \
  --image-url "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg" \
  --prompt "Describe the image." \
  --max-tokens 128 \
  --temp 0.0
```

NLLB:

```bash
python examples/translation/nllb_translation.py \
  --model_name facebook/nllb-200-distilled-600M \
  --revision refs/pr/45 \
  --source_language English \
  --target_language Yoruba \
  --text_to_translate "Let us translate text to Yoruba"
```

Chat UI:

```bash
cd chat
bash start.sh
```

Benchmark:

```bash
python examples/text_generation/benchmark_generation.py --help
```

## Tests

The default suite is bounded and does not download Hub models:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m unittest discover -s tests -v
```

Tests that use external checkpoints are skipped unless
`MLX_TRANSFORMERS_RUN_HUB_TESTS=1` is set. Review their model IDs and expected
download sizes before opting in. Some checkpoints are gated; set `HF_TOKEN`
only for an explicitly reviewed integration run.

The verified 2026-07-26 Apple-silicon baseline is 113 discovered tests:
91 pass and 22 Hub integration tests skip.
