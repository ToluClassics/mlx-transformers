import os
import unittest

import mlx.core as mx
import numpy as np
from transformers import (
    AutoTokenizer,
    RobertaConfig,
    RobertaForQuestionAnswering,
    RobertaForSequenceClassification,
    RobertaForTokenClassification,
    RobertaModel,
    RobertaTokenizer,
)

from src.mlx_transformers.models import (
    RobertaForQuestionAnswering as MlxRobertaForQuestionAnswering,
)
from src.mlx_transformers.models import (
    RobertaForSequenceClassification as MlxRobertaForSequenceClassification,
)
from src.mlx_transformers.models import (
    RobertaForTokenClassification as MlxRobertaForTokenClassification,
)
from src.mlx_transformers.models import RobertaModel as MlxRobertaModel


def load_hgf_model(model_name: str, hgf_model_class):
    model = hgf_model_class.from_pretrained(model_name)
    return model


class TestMlxRobertaForSequenceClassification(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "cardiffnlp/twitter-roberta-base-emotion"
        cls.hgf_model_class = RobertaForSequenceClassification
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_text = "Hello, my dog is cute"

        cls.config = RobertaConfig.from_pretrained(cls.model_name)
        cls.model = MlxRobertaForSequenceClassification(cls.config)
        cls.model.from_pretrained(cls.model_name, revision="refs/pr/3")

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
        cls.hgf_model_class = RobertaForTokenClassification
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_text = "HuggingFace is a company based in Paris and New York"

        cls.config = RobertaConfig.from_pretrained(cls.model_name)
        cls.model = MlxRobertaForTokenClassification(cls.config)
        cls.model.from_pretrained(cls.model_name)

    def test_forward(self) -> None:
        inputs = self.tokenizer(
            self.input_text,
            return_tensors="np",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsInstance(outputs.logits, mx.array)

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


class TestMlxRobertaForQuestionAnswering(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "deepset/roberta-base-squad2"
        cls.hgf_model_class = RobertaForQuestionAnswering
        cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        cls.input_question = "Who was Jim Henson?"
        cls.input_text = "Jim Henson was a nice puppet"

        cls.config = RobertaConfig.from_pretrained(cls.model_name)
        cls.model = MlxRobertaForQuestionAnswering(cls.config)
        cls.model.from_pretrained(cls.model_name)

    def test_forward(self) -> None:
        inputs = self.tokenizer(
            self.input_question, self.input_text, return_tensors="np"
        )

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs)
        self.assertIsInstance(outputs.start_logits, mx.array)
        self.assertIsInstance(outputs.end_logits, mx.array)

    def test_model_output_hgf(self):
        inputs_mlx = self.tokenizer(
            self.input_question, self.input_text, return_tensors="np"
        )

        inputs_mlx = {key: mx.array(v) for key, v in inputs_mlx.items()}
        outputs_mlx = self.model(**inputs_mlx)

        mlx_answer_start_index = outputs_mlx.start_logits.argmax().item()
        mlx_answer_end_index = outputs_mlx.end_logits.argmax().item()
        mlx_predict_answer_tokens = inputs_mlx["input_ids"].tolist()
        mlx_predict_answer_tokens = mlx_predict_answer_tokens[0][
            mlx_answer_start_index : mlx_answer_end_index + 1
        ]
        mlx_answer = self.tokenizer.decode(
            mlx_predict_answer_tokens, skip_special_tokens=True
        )

        inputs_hgf = self.tokenizer(
            self.input_question, self.input_text, return_tensors="pt"
        )

        hgf_model = load_hgf_model(self.model_name, self.hgf_model_class)
        outputs_hgf = hgf_model(**inputs_hgf)

        hgf_answer_start_index = outputs_hgf.start_logits.argmax()
        hgf_answer_end_index = outputs_hgf.end_logits.argmax()
        hgf_predict_answer_tokens = inputs_hgf.input_ids[
            0, hgf_answer_start_index : hgf_answer_end_index + 1
        ]
        hgf_answer = self.tokenizer.decode(
            hgf_predict_answer_tokens, skip_special_tokens=True
        )

        self.assertEqual(mlx_answer, hgf_answer)


if __name__ == "__main__":
    unittest.main()
