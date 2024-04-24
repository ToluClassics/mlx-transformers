import os
import unittest

import mlx.core as mx
import numpy as np
from transformers import XLMRobertaConfig, XLMRobertaModel, XLMRobertaTokenizer

from src.mlx_transformers.models import XLMRobertaModel as MlxXLMRobertaModel
from src.mlx_transformers.models.utils import convert


def load_model(model_name: str) -> MlxXLMRobertaModel:
    current_directory = os.path.dirname(os.path.realpath(__file__))
    weights_path = os.path.join(
        current_directory, "model_checkpoints", model_name.replace("/", "-") + ".npz"
    )

    if not os.path.exists(weights_path):
        convert(model_name, weights_path, XLMRobertaModel)

    config = XLMRobertaConfig.from_pretrained(model_name)
    model = MlxXLMRobertaModel(config)

    model.load_weights(weights_path, strict=True)

    return model


def load_hgf_model(model_name: str) -> XLMRobertaModel:
    model = XLMRobertaModel.from_pretrained(model_name)
    return model


class TestMlxXLMRoberta(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model_name = "FacebookAI/xlm-roberta-base"
        cls.model = load_model(cls.model_name)
        cls.tokenizer = XLMRobertaTokenizer.from_pretrained(cls.model_name)
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
        hgf_model = load_hgf_model(self.model_name)
        outputs_hgf = hgf_model(**inputs_hgf)
        outputs_hgf = outputs_hgf.last_hidden_state.detach().numpy()

        self.assertTrue(np.allclose(outputs_mlx, outputs_hgf, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
