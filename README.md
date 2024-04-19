# MLX Transformers

[![PyPI](https://img.shields.io/pypi/v/mlx-transformers?color=red)](https://pypi.org/project/mlx-transformers/)


MLX Transformers is a library that provides model implementation in [MLX](https://github.com/ml-explore/mlx). It uses a similar model interface as HuggingFace Transformers and provides a way to load and use models in Apple Silicon devices. Implemented models have the same modules 

MLX transformers is currently only available for infernce on Apple Silicon devices. Training support will be added in the future.

# Installation

This library is available on PyPI and can be installed using pip:

```bash
pip install mlx-transformers
```


## Quick Tour

A list of the available models can be found in the `mlx_transformers.models` module and are also listed in the [section below](#available-model-architectures). The following example demonstrates how to load a model and use it for inference:

- First you need to download and convert the model checkpoint to MLX format
    To do this from huggingface

    ```python

    from transformers import BertModel
    from mlx_transformers.models.utils import convert

    model_name_or_path = "bert-base-uncased"
    mlx_checkpoint = "bert-base-uncased.npz"

    convert("bert-base-uncased", "bert-base-uncased.npz", BertModel)
    ```
    This will download the model checkpoint from huggingface and convert it to MLX format.

- Now you can load the model using MLX transformers in few lines of code

    ```python
    from transformers import BertConfig, BertTokenizer
    from mlx_transformers.models import BertModel as MLXBertModel

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    config = BertConfig.from_pretrained("bert-base-uncased")
    model = MLXBertModel(config)

    model.load_weights("bert-base-uncased.npz", strict=True)

    sample_input = "Hello, world!"
    inputs = tokenizer(sample_input, return_tensors="np")
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
model.load_weights("all-MiniLM-L6-v2.npz", strict=True)

inputs = tokenizer(sentences, return_tensors="np", padding=True, truncation=True)
outputs = model(**inputs)

sentence_embeddings = _mean_pooling(outputs.last_hidden_state, inputs.attention_mask)
```


## Available Models

The following models have been ported to MLX Transformers from Huggingface for inference:

1. Bert
2. Roberta
3. XLMRoberta
4. M2M100
5. Sentence Transformers

## Examples

Coming soon...

## Benchmarks

Coming soon...

## Contributions

Contributions to MLX transformers are welcome. See the contributing documentation for instructions on setting up a development environment.