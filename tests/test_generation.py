import unittest
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from src.mlx_transformers.models.base import MlxPretrainedMixin


class FakeGenerationModel(MlxPretrainedMixin):
    def __init__(self):
        self.config = SimpleNamespace(eos_token_id=9, pad_token_id=0)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if len(self.calls) == 1:
            logits = np.full((2, 3, 10), -100.0, dtype=np.float32)
            logits[0, 1, 9] = 100.0
            logits[0, 2, 7] = 100.0
            logits[1, 2, 4] = 100.0
        else:
            logits = np.full((2, 1, 10), -100.0, dtype=np.float32)
            logits[0, 0, 3] = 100.0
            logits[1, 0, 9] = 100.0

        return SimpleNamespace(
            logits=mx.array(logits),
            past_key_values=f"cache-{len(self.calls)}",
        )

    def prepare_inputs_for_generation(self, **kwargs):
        return kwargs


class TestGenerationContract(unittest.TestCase):
    def test_batched_generation_handles_padding_eos_and_input_immutability(self):
        model = FakeGenerationModel()
        input_ids = mx.array([[1, 2, 0], [0, 3, 4]], dtype=mx.int32)
        attention_mask = mx.array([[1, 1, 0], [0, 1, 1]], dtype=mx.int32)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        tokens = list(
            model._generate_tokens(
                inputs,
                max_new_tokens=5,
                temp=0.0,
            )
        )

        self.assertEqual(len(tokens), 2)
        np.testing.assert_array_equal(np.array(tokens[0]), [9, 4])
        np.testing.assert_array_equal(np.array(tokens[1]), [0, 9])
        np.testing.assert_array_equal(
            np.array(inputs["input_ids"]), [[1, 2, 0], [0, 3, 4]]
        )
        np.testing.assert_array_equal(
            np.array(inputs["attention_mask"]),
            [[1, 1, 0], [0, 1, 1]],
        )
        self.assertEqual(model.calls[1]["input_ids"].shape, (2, 4))
        self.assertEqual(model.calls[1]["attention_mask"].shape, (2, 4))
        self.assertEqual(model.calls[1]["past_key_values"], "cache-1")

    def test_generation_requires_a_finite_bound(self):
        model = FakeGenerationModel()
        inputs = {"input_ids": mx.array([[1]], dtype=mx.int32)}

        with self.assertRaisesRegex(ValueError, "max_new_tokens is required"):
            list(model._generate_tokens(inputs, temp=0.0))

    def test_zero_new_tokens_does_not_run_the_model(self):
        model = FakeGenerationModel()
        inputs = {"input_ids": mx.array([[1]], dtype=mx.int32)}

        self.assertEqual(
            list(model._generate_tokens(inputs, max_new_tokens=0)),
            [],
        )
        self.assertEqual(model.calls, [])
