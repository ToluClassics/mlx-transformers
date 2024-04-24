import unittest

import mlx.core as mx
from transformers import AutoTokenizer, PhiConfig, PhiForCausalLM

from src.mlx_transformers.models import PhiForCausalLM as MlxPhiForCausalLM


def load_hgf_model(model_name: str) -> PhiForCausalLM:
    model = PhiForCausalLM.from_pretrained(model_name)
    return model


class TestMlxPhi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "microsoft/phi-2"
        config = PhiConfig.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxPhiForCausalLM(config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_text = "Hey, are you conscious? Can you talk to me?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.logits) == mx.array
