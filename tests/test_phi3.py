import unittest

import mlx.core as mx
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

from src.mlx_transformers.models import Phi3ForCausalLM as MlxPhi3ForCausalLM


def load_hgf_model(model_name: str) -> AutoModelForCausalLM:
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model


class TestMlxPhi3(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "microsoft/Phi-3-mini-4k-instruct"
        config = AutoConfig.from_pretrained(cls.model_name)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxPhi3ForCausalLM(config)
        cls.model.from_pretrained(
            cls.model_name,
            huggingface_model_architecture="AutoModelForCausalLM",
            trust_remote_code=True,
        )

        cls.input_text = "Hey, are you conscious? Can you talk to me?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.logits) == mx.array
