# MLX Transformers

[![PyPI](https://img.shields.io/pypi/v/mlx-transformers?color=red)](https://pypi.org/project/mlx-transformers/)

MLX implementations of Hugging Face-style models for Apple Silicon.

## Installation

```bash
pip install mlx-transformers
```

For local development:

```bash
pip install -r requirements.txt
pip install -e .
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

Quantized loading:

```python
model.from_pretrained(
    model_name,
    quantize=True,
    group_size=64,
    bits=4,
    mode="affine",
)
```

## Supported Models

- BERT
- RoBERTa
- XLM-RoBERTa
- LLaMA
- Phi
- Phi-3
- Qwen3
- Qwen3-VL
- OpenELM
- Persimmon
- Fuyu
- Gemma3
- M2M100 / NLLB

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

```bash
python -m unittest
```

Some models are gated on Hugging Face. Set `HF_TOKEN` if needed.
