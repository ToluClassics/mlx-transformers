# MLX Transformers

[![PyPI](https://img.shields.io/pypi/v/mlx-transformers?color=red)](https://pypi.org/project/mlx-transformers/)


MLX Transformers is a library that provides model implementations in [MLX](https://github.com/ml-explore/mlx). It uses a similar model interface as HuggingFace Transformers and provides a way to load and use models in Apple Silicon devices. Implemented models have the same modules and module key as the original implementations in transformers.

MLX transformers is currently only available for inference on Apple Silicon devices. Training support will be added in the future.

# Installation

This library is available on PyPI and can be installed using pip:

```bash
pip install mlx-transformers
```


## Quick Tour

A list of the available models can be found in the `mlx_transformers.models` module and are also listed in the [section below](#available-model-architectures). The following example demonstrates how to load a model and use it for inference:


- You can load the model using MLX transformers in few lines of code

    ```python
    import mlx.core as mx
    from transformers import BertConfig, BertTokenizer
    from mlx_transformers.models import BertForMaskedLM as MLXBertForMaskedLM

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_pretrained("bert-base-uncased")
    
    model = MLXBertForMaskedLM(config)
    model.from_pretrained("bert-base-uncased")

    sample_input = "Hello, world!"
    inputs = tokenizer(sample_input, return_tensors="np")
    inputs = {key: mx.array(v) for key, v in inputs.items()}

    outputs = model(**inputs)
    ```

### Sentence Transformer Example

```python
import mlx.core as mx
import numpy as np

from transformers import AutoConfig, AutoTokenizer
from mlx_transformers.models import BertModel as MLXBertModel


def _mean_pooling(last_hidden_state: mx.array, attention_mask: mx.array):
    token_embeddings = last_hidden_state
    input_mask_expanded = mx.expand_dims(attention_mask, -1)
    input_mask_expanded = mx.broadcast_to(input_mask_expanded, token_embeddings.shape).astype(mx.float32)
    sum_embeddings = mx.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = mx.clip(input_mask_expanded.sum(axis=1), 1e-9, None)
    return sum_embeddings / sum_mask

sentences = ['This is an example sentence', 'Each sentence is converted']

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
config = AutoConfig.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

model = MLXBertModel(config)
model.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

inputs = tokenizer(sentences, return_tensors="np", padding=True, truncation=True)
inputs = {key: mx.array(v) for key, v in inputs.items()}

outputs = model(**inputs)

sentence_embeddings = _mean_pooling(outputs.last_hidden_state, inputs.attention_mask)
```


## Examples

The `examples` directory contains a few examples that demonstrate how to use the models in MLX Transformers. 

1. [LLama Example](examples/llama_generation.py)
    ```bash
    python3 examples/llama_generation.py --model-name "meta-llama/Llama-2-7b-hf"  
    ```

2. [NLLB Translation Example](examples/nllb_translation.py)
    ```bash
    python3 examples/nllb_translation.py --model_name facebook/nllb-200-distilled-600M --source_language English --target_language Yoruba --text_to_translate "Let us translate text to Yoruba"

    Output:==> ['Ẹ jẹ́ ká tú àwọn ẹsẹ Bíbélì sí èdè Yoruba']
    ```


## Benchmarks

Coming soon...

## Contributions

Contributions to MLX transformers are welcome. We would like to have as many model implementations as possible.
See the contributing documentation for instructions on setting up a development environment.