import os
import sys
import unittest

import mlx.core as mx
import numpy as np
from transformers import (
    RobertaConfig,
    RobertaModel,
    RobertaTokenizer,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaForQuestionAnswering,
    AutoTokenizer
) 

from src.mlx_transformers.models import RobertaModel as MlxRobertaModel
from src.mlx_transformers.models import RobertaForSequenceClassification as MlxRobertaForSequenceClassification
from src.mlx_transformers.models import RobertaForTokenClassification as MlxRobertaForTokenClassification
from src.mlx_transformers.models import RobertaForQuestionAnswering as MlxRobertaForQuestionAnswering
from src.mlx_transformers.models.utils import convert


def load_model(model_name: str, model_class, hgf_model_class):
    current_directory = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(
        current_directory, "model_checkpoints", model_name.replace("/", "-") + ".npz"
    )

    if not os.path.exists(weights_path):
        convert(model_name, weights_path, hgf_model_class)

    config = RobertaConfig.from_pretrained(model_name)
    # print(config)
    model = model_class(config)

    model.load_weights(weights_path, strict=True)

    return model


def load_hgf_model(model_name: str, hgf_model_class):
    model = hgf_model_class.from_pretrained(model_name)
    return model


class TestMlxRoberta(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "FacebookAI/roberta-base"
        cls.model_class = MlxRobertaModel
        cls.hgf_model_class = RobertaModel
        cls.model = load_model(cls.model_name, cls.model_class, cls.hgf_model_class)
        cls.tokenizer = RobertaTokenizer.from_pretrained(cls.model_name)
        cls.input_text = "Hello, my dog is cute"

    def test_forward(self) -> None:
        inputs = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True
        )

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsInstance(outputs.last_hidden_state, mx.array)

    def test_model_output_hgf(self):
        inputs_mlx = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True
        )

        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        outputs_mlx = self.model(**inputs_mlx)
        outputs_mlx = np.array(outputs_mlx.last_hidden_state)

        inputs_hgf = self.tokenizer(
            self.input_text, return_tensors="pt", padding=True, truncation=True
        )
        hgf_model = load_hgf_model(self.model_name, self.hgf_model_class)
        outputs_hgf = hgf_model(**inputs_hgf)
        outputs_hgf = outputs_hgf.last_hidden_state.detach().numpy()

        self.assertTrue(np.allclose(outputs_mlx, outputs_hgf, atol=1e-4))

class TestMlxRobertaForSequenceClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "cardiffnlp/twitter-roberta-base-emotion"
        cls.model_class = MlxRobertaForSequenceClassification
        cls.hgf_model_class = RobertaForSequenceClassification
        cls.model = load_model(cls.model_name, cls.model_class, cls.hgf_model_class)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_text = "Hello, my dog is cute"


    def test_forward(self) -> None:
        inputs = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True
        )

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsInstance(outputs.logits, mx.array)

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


class TestMlxRobertaForTokenClassification(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "Jean-Baptiste/roberta-large-ner-english"
        cls.model_class = MlxRobertaForTokenClassification
        cls.hgf_model_class = RobertaForTokenClassification
        cls.model = load_model(cls.model_name, cls.model_class, cls.hgf_model_class)
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_text = "HuggingFace is a company based in Paris and New York"


    def test_forward(self) -> None:
        inputs = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True, add_special_tokens=False
        )

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsInstance(outputs.logits, mx.array)

    def test_model_output_hgf(self):
        inputs_mlx = self.tokenizer(
            self.input_text, return_tensors="np", padding=True, truncation=True, add_special_tokens=False
        )

        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        outputs_mlx = self.model(**inputs_mlx)
        outputs_mlx = np.array(outputs_mlx.logits)
        mlx_predicted_token_class_ids = outputs_mlx.argmax(-1)
        mlx_predicted_tokens_classes = [self.model.config.id2label[t.item()] for t in mlx_predicted_token_class_ids[0]]


        inputs_hgf = self.tokenizer(
            self.input_text, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
        )
        hgf_model = load_hgf_model(self.model_name, self.hgf_model_class)
        outputs_hgf = hgf_model(**inputs_hgf)
        outputs_hgf = outputs_hgf.logits

        hgf_predicted_token_class_ids = outputs_hgf.argmax(-1)
        hgf_predicted_tokens_classes = [hgf_model.config.id2label[t.item()] for t in hgf_predicted_token_class_ids[0]]

        self.assertEqual(mlx_predicted_tokens_classes, hgf_predicted_tokens_classes)



if __name__ == "__main__":
    unittest.main()
