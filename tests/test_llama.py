import unittest

import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from src.mlx_transformers.models import LlamaForCausalLM as MlxLlamaForCausalLM


def load_hgf_model(model_name: str) -> LlamaForCausalLM:
    model = LlamaForCausalLM.from_pretrained(model_name)
    return model


class TestMlxLlama(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "meta-llama/Llama-3.2-1B"
        config = LlamaConfig.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxLlamaForCausalLM(config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_text = "Who is the prince of Wales?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.logits) == mx.array

        hgf_model = load_hgf_model(self.model_name)
        hgf_inputs = self.tokenizer(self.input_text, return_tensors="pt", truncation=True)

        hgf_outputs = hgf_model(**hgf_inputs, use_cache=True)

        self.assertTrue(
            np.allclose(np.array(outputs.logits), hgf_outputs.logits.detach().numpy(), atol=1e-4)
        )





