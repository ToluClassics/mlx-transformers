import unittest
import numpy as np
import mlx.core as mx
from transformers import (
    AutoTokenizer,
    PersimmonConfig,
    PersimmonForCausalLM,
    PersimmonForSequenceClassification,
)

from src.mlx_transformers.models import PersimmonForCausalLM as MlxPersimmonForCausalLM
from src.mlx_transformers.models import PersimmonForSequenceClassification as MlxPersimmonForSequenceClassification


def load_hgf_model(model_name: str, hgf_model_class: str):
    model = hgf_model_class.from_pretrained(model_name)
    return model


class MlxPersimmon(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "adept/persimmon-8b-base"
        config = PersimmonConfig.from_pretrained(cls.model_name)
        cls.hgf_model_class = PersimmonForCausalLM
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxPersimmonForCausalLM(config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_text = "human: Hey, what should I eat for dinner?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.logits) == mx.array


class TestMlxPersimmonForSequenceClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "adept/persimmon-8b-base"
        config = PersimmonConfig.from_pretrained(cls.model_name)
        cls.hgf_model_class = PersimmonForSequenceClassification
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxPersimmonForSequenceClassification(config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_text = "human: Hey, what should I eat for dinner?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.logits) == mx.array

    def test_model_output_hgf(self):
        inputs_mlx = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True
        )

        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        outputs_mlx = self.model(**inputs_mlx)
        outputs_mlx = np.array(outputs_mlx.logits)
        predicted_class_id = outputs_mlx.argmax().item()
        mlx_label = self.model.config.id2label[predicted_class_id]

        inputs_hgf = self.tokenizer(
            self.input_text, return_tensors="pt", padding=True, truncation=True
        )
        hgf_model = load_hgf_model(self.model_name, self.hgf_model_class)
        outputs_hgf = hgf_model(**inputs_hgf)
        outputs_hgf = outputs_hgf.logits

        predicted_class_id = outputs_hgf.argmax().item()
        hgf_label = hgf_model.config.id2label[predicted_class_id]

        self.assertEqual(mlx_label, hgf_label)
