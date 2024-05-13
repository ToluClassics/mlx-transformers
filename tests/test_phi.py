import unittest
import numpy as np
import mlx.core as mx
from transformers import (
    AutoTokenizer,
    PhiConfig,
    PhiForCausalLM,
    PhiForSequenceClassification,
    PhiForTokenClassification,
)

from src.mlx_transformers.models import PhiForCausalLM as MlxPhiForCausalLM
from src.mlx_transformers.models import (
    PhiForSequenceClassification as MlxPhiForSequenceClassification,
)
from src.mlx_transformers.models import (
    PhiForTokenClassification as MlxPhiForTokenClassification,
)


def load_hgf_model(model_name: str, hgf_model_class: str):
    model = hgf_model_class.from_pretrained(model_name)
    return model


class TestMlxPhi(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "microsoft/phi-2"
        config = PhiConfig.from_pretrained(cls.model_name)
        cls.hgf_model_class = PhiForCausalLM
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxPhiForCausalLM(config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_text = "Hey, are you conscious? Can you talk to me?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.logits) == mx.array


class TestMlxPhiForTokenClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "microsoft/phi-2"
        config = PhiConfig.from_pretrained(cls.model_name)
        cls.hgf_model_class = PhiForTokenClassification
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxPhiForTokenClassification(config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_text = "Hey, are you conscious? Can you talk to me?"

    def test_forward(self) -> None:
        inputs = self.tokenizer(self.input_text, return_tensors="np", truncation=True)

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=True)

        assert type(outputs.logits) == mx.array

    def test_model_output_hgf(self):
        inputs_mlx = self.tokenizer(
            self.input_text,
            return_tensors="np",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )

        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        outputs_mlx = self.model(**inputs_mlx)
        outputs_mlx = np.array(outputs_mlx.logits)
        mlx_predicted_token_class_ids = outputs_mlx.argmax(-1)
        mlx_predicted_tokens_classes = [
            self.model.config.id2label[t.item()]
            for t in mlx_predicted_token_class_ids[0]
        ]

        inputs_hgf = self.tokenizer(
            self.input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )
        hgf_model = load_hgf_model(self.model_name, self.hgf_model_class)
        outputs_hgf = hgf_model(**inputs_hgf)
        outputs_hgf = outputs_hgf.logits

        hgf_predicted_token_class_ids = outputs_hgf.argmax(-1)
        hgf_predicted_tokens_classes = [
            hgf_model.config.id2label[t.item()]
            for t in hgf_predicted_token_class_ids[0]
        ]

        self.assertEqual(mlx_predicted_tokens_classes, hgf_predicted_tokens_classes)


class TestMlxPhiForSequenceClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "microsoft/phi-2"
        config = PhiConfig.from_pretrained(cls.model_name)
        cls.hgf_model_class = PhiForSequenceClassification
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.model = MlxPhiForSequenceClassification(config)
        cls.model.from_pretrained(cls.model_name)

        cls.input_text = "Hey, are you conscious? Can you talk to me?"

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
