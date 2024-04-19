# MLX Transformers

[![PyPI](https://img.shields.io/pypi/v/mlx-transformers?color=red)](https://pypi.org/project/mlx-transformers/)


MLX Transformers is a library that provides model implementation in MLX. It uses a similar model interface as HuggingFace Transformers and provides a way to load and use models in Apple Silicon devices. Implemented models have the same modules 

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

    sample_input = "Hello, world!"
    inputs = tokenizer(sample_input, return_tensors="np")
    outputs = model(**inputs)
    ```


## Available Models

The following models have been ported to MLX Transformers from Huggingface for inference:

1. Bert
2. Roberta
3. XLMRoberta
4. M2M100

## Examples

Coming soon...

## Benchmarks

Coming soon...

## Contributions

Contributions to MLX transformers are welcome. See the contributing documentation for instructions on setting up a development environment.