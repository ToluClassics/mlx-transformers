import os
import sys
import unittest

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from src.mlx_transformers.models import LlamaForCausalLM as MlxLlamaForCausalLM
from src.mlx_transformers.models.utils import convert


def load_model(model_name: str, config, hgf_model_class, mlx_model_class):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(
        current_directory, "model_checkpoints", model_name.replace("/", "-") + ".npz"
    )

    if not os.path.exists(weights_path):
        convert(model_name, weights_path, hgf_model_class)

    model = mlx_model_class(config)

    print(model)

    model.load_weights(weights_path, strict=False)
    return model


def load_hgf_model(model_name: str) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(model_name)
    return model


class TestMlxLlama(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "meta-llama/Llama-2-7b-hf"
        config = LlamaConfig.from_pretrained(cls.model_name)

        cls.model = load_model(
            cls.model_name, config, LlamaForCausalLM, MlxLlamaForCausalLM
        )
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_text = "Hey, are you conscious? Can you talk to me?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.last_hidden_state) == mx.array
