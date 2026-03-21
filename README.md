# MLX Transformers

[![PyPI](https://img.shields.io/pypi/v/mlx-transformers?color=red)](https://pypi.org/project/mlx-transformers/)

`mlx-transformers` provides MLX implementations of several Hugging Face-style model architectures for Apple Silicon. The project keeps a familiar Transformers-style API while loading weights from Hugging Face checkpoints and running inference with MLX.

The repository is currently inference-focused. Some model families have broader parity than others, but the core usage pattern is the same across the package:

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

## Requirements

- Apple Silicon Mac
- Python 3.10+
- MLX-compatible environment

Some models are gated on Hugging Face. If needed, set `HF_TOKEN` in your environment before calling `from_pretrained(...)`.

## Installation

Install from PyPI:

```bash
pip install mlx-transformers
```

Install for local development:

```bash
pip install -r requirements.txt
pip install -e .
```

`asitop` is also useful if you want to monitor GPU and CPU usage on Apple Silicon:

```bash
pip install asitop
```

## Available Models

Current exports from `src/mlx_transformers/models/__init__.py`:

- BERT
  - `BertModel`
  - `BertForMaskedLM`
  - `BertForSequenceClassification`
  - `BertForTokenClassification`
  - `BertForQuestionAnswering`
- RoBERTa
  - `RobertaModel`
  - `RobertaForSequenceClassification`
  - `RobertaForTokenClassification`
  - `RobertaForQuestionAnswering`
- XLM-RoBERTa
  - `XLMRobertaModel`
  - `XLMRobertaForSequenceClassification`
  - `XLMRobertaForTokenClassification`
  - `XLMRobertaForQuestionAnswering`
- Causal LMs
  - `LlamaModel`, `LlamaForCausalLM`
  - `PhiModel`, `PhiForCausalLM`
  - `Phi3Model`, `Phi3ForCausalLM`
  - `OpenELMModel`, `OpenELMForCausalLM`
  - `PersimmonForCausalLM`
  - `FuyuForCausalLM`
- Translation
  - `M2M100ForConditionalGeneration`

## Examples

### Sentence Embeddings with BERT

```bash
python examples/bert/sentence_transformers.py
```

### LLaMA Text Generation

The LLaMA example now formats the input with the tokenizer chat template and stops on EOS.

```bash
python examples/text_generation/llama_generation.py \
  --model-name meta-llama/Llama-3.2-1B-Instruct \
  --prompt "Write a short explanation of rotary embeddings." \
  --max-tokens 128 \
  --temp 0.0
```

### Phi-3 Text Generation

```bash
python examples/text_generation/phi3_generation.py \
  --model-name microsoft/Phi-3-mini-4k-instruct \
  --prompt "Explain attention masking." \
  --max-tokens 128 \
  --temp 0.0
```

### OpenELM Text Generation

```bash
python examples/text_generation/openelm_generation.py \
  --model-name apple/OpenELM-1_1B-Instruct \
  --prompt "Summarize grouped-query attention." \
  --max-tokens 128
```

### NLLB / M2M-100 Translation

```bash
python examples/translation/nllb_translation.py \
  --model_name facebook/nllb-200-distilled-600M \
  --source_language English \
  --target_language Yoruba \
  --text_to_translate "Let us translate text to Yoruba"
```

## Chat Interface

A Streamlit chat UI is included under `chat/`.

```bash
cd chat
bash start.sh
```

![Chat Image](images/mlx_transformer_chat.png)

## Tests

The repository currently includes focused tests for:

- BERT
- RoBERTa
- XLM-RoBERTa
- LLaMA
- Phi
- Phi-3

Run the full test suite:

```bash
python -m unittest
```

Run a single module:

```bash
python -m unittest tests.test_bert
python -m unittest tests.test_llama
```

Some tests download model weights from Hugging Face on first run.

## Repository Layout

```text
src/mlx_transformers/models/   model implementations and shared helpers
examples/                      runnable examples
tests/                         model parity and behavior tests
chat/                          streamlit chat interface
```

## Notes

- Model loading is handled through `from_pretrained(...)` in `src/mlx_transformers/models/base.py`.
- Pretrained models are loaded in eval mode by default.
- Causal generation support is present for the decoder-style model families, but parity and feature coverage still vary by architecture.

## Contributing

Contributions are welcome. The highest-value contributions are usually:

- new model implementations
- parity fixes against Hugging Face behavior
- generation and cache correctness fixes
- tests for unsupported or weakly covered paths
