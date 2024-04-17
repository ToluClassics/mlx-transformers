import os
import sys
import unittest

import mlx.core as mx
import numpy
from transformers import BertConfig, BertModel, BertTokenizer

from mlx_transformers.models.bert import BertModel as MlxBertModel


def convert(model_name: str, mlx_model: str) -> None:
    model = BertModel.from_pretrained(model_name)
    # save the tensors
    tensors = {key: tensor.numpy() for key, tensor in model.state_dict().items()}
    numpy.savez(mlx_model, **tensors)


def load_model(model_name: str) -> MlxBertModel:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(
        current_directory, "model_checkpoints", model_name.replace("/", "-") + ".npz"
    )

    if not os.path.exists(weights_path):
        convert(model_name, weights_path)

    config = BertConfig.from_pretrained(model_name)
    model = MlxBertModel(config)

    model.load_weights(weights_path, strict=True)

    return model


def load_hgf_model(model_name: str) -> BertModel:
    model = BertModel.from_pretrained(model_name)
    return model


class TestMlxBert(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "bert-base-uncased"
        cls.model = load_model(cls.model_name)
        cls.tokenizer = BertTokenizer.from_pretrained(cls.model_name)

    def test_forward(self) -> None:
        input_text = "Hello, my dog is cute"
        inputs = self.tokenizer(
            input_text, return_tensors="np", padding=True, truncation=True
        )

        inputs = {key: mx.array(v) for key, v in inputs.items()}
        outputs = self.model(**inputs)

        self.assertIsInstance(outputs.last_hidden_state, mx.array)

    def test_model_output_hgf(self):
        pass


if __name__ == "__main__":
    unittest.main()
