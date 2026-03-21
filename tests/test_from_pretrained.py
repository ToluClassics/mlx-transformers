import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.mlx_transformers.models.base import MlxPretrainedMixin


class DummyTensor:
    def __init__(self):
        self.dtype = None

    def astype(self, dtype):
        self.dtype = dtype
        return self


class DummyModel(MlxPretrainedMixin):
    def __init__(self):
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.updated_weights = None
        self.loaded_weights = None
        self.eval_called = False

    def parameters(self):
        return {"layer.weight": object()}

    def load_weights(self, weights, strict=False):
        self.loaded_weights = (dict(weights), strict)

    def update(self, weights):
        self.updated_weights = weights

    def eval(self):
        self.eval_called = True


class FakeLinear:
    pass


class FakeEmbedding:
    pass


class FakeQuantizableModule:
    def to_quantized(self, *args, **kwargs):
        return self


class TestFromPretrainedQuantization(unittest.TestCase):
    def test_from_pretrained_quantizes_loaded_model(self):
        tensor = DummyTensor()
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()

            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={"layer.weight": tensor},
            ), patch(
                "src.mlx_transformers.models.base.tree_flatten",
                return_value=[("layer.weight", object())],
            ), patch(
                "src.mlx_transformers.models.base.tree_unflatten",
                side_effect=lambda items: dict(items),
            ), patch(
                "src.mlx_transformers.models.base.nn.quantize",
            ) as mock_quantize:
                result = model.from_pretrained(
                    tmpdir,
                    quantize=True,
                    group_size=64,
                    bits=4,
                )

        self.assertIs(result, model)
        self.assertEqual(model.loaded_weights, ({"layer.weight": tensor}, False))
        self.assertIsNone(model.updated_weights)
        self.assertTrue(model.eval_called)
        mock_quantize.assert_called_once_with(
            model,
            group_size=64,
            bits=4,
            mode="affine",
            quantize_input=False,
            class_predicate=None,
        )

    def test_from_pretrained_restricts_quantize_input_to_linear_layers(self):
        tensor = DummyTensor()
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()

            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={"layer.weight": tensor},
            ), patch(
                "src.mlx_transformers.models.base.tree_flatten",
                return_value=[("layer.weight", object())],
            ), patch(
                "src.mlx_transformers.models.base.tree_unflatten",
                side_effect=lambda items: dict(items),
            ), patch(
                "src.mlx_transformers.models.base.nn.Linear",
                FakeLinear,
            ), patch(
                "src.mlx_transformers.models.base.nn.quantize",
            ) as mock_quantize:
                model.from_pretrained(
                    tmpdir,
                    quantize=True,
                    mode="nvfp4",
                    quantize_input=True,
                )

        kwargs = mock_quantize.call_args.kwargs
        predicate = kwargs["class_predicate"]

        self.assertTrue(predicate("model.layers.0", FakeLinear()))
        self.assertFalse(predicate("model.embed_tokens", FakeEmbedding()))

    def test_from_pretrained_validates_quantize_input_mode(self):
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()

            with self.assertRaisesRegex(
                ValueError,
                "quantize_input=True is only supported",
            ):
                model.from_pretrained(
                    tmpdir,
                    quantize=True,
                    mode="affine",
                    quantize_input=True,
                )

    def test_from_pretrained_loads_prequantized_mlx_checkpoint(self):
        quantized_weight = DummyTensor()
        quantized_scales = DummyTensor()
        quantized_biases = DummyTensor()
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            Path(tmpdir, "config.json").write_text(
                json.dumps({"quantization": {"group_size": 64, "bits": 4}}),
                encoding="utf-8",
            )

            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={
                    "layer.weight": quantized_weight,
                    "layer.scales": quantized_scales,
                    "layer.biases": quantized_biases,
                },
            ), patch(
                "src.mlx_transformers.models.base.tree_flatten",
                return_value=[
                    ("layer.weight", object()),
                    ("layer.scales", object()),
                    ("layer.biases", object()),
                ],
            ), patch(
                "src.mlx_transformers.models.base.tree_unflatten",
                side_effect=lambda items: dict(items),
            ), patch(
                "src.mlx_transformers.models.base.nn.quantize",
            ) as mock_quantize:
                model.from_pretrained(tmpdir)

        self.assertEqual(model.loaded_weights[0]["layer.weight"], quantized_weight)
        self.assertEqual(model.loaded_weights[0]["layer.scales"], quantized_scales)
        self.assertEqual(model.loaded_weights[0]["layer.biases"], quantized_biases)
        self.assertFalse(model.loaded_weights[1])
        self.assertEqual(model.config.quantization, {"group_size": 64, "bits": 4})
        self.assertIsNone(quantized_weight.dtype)
        self.assertIsNone(quantized_scales.dtype)
        self.assertIsNone(quantized_biases.dtype)

        kwargs = mock_quantize.call_args.kwargs
        predicate = kwargs["class_predicate"]

        self.assertEqual(kwargs["group_size"], 64)
        self.assertEqual(kwargs["bits"], 4)
        self.assertEqual(kwargs["mode"], "affine")
        self.assertTrue(predicate("layer", FakeQuantizableModule()))
        self.assertFalse(predicate("other", FakeQuantizableModule()))
        self.assertFalse(predicate("layer", object()))

    def test_from_pretrained_rejects_requantizing_prequantized_checkpoint(self):
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            Path(tmpdir, "config.json").write_text(
                json.dumps({"quantization": {"group_size": 64, "bits": 4}}),
                encoding="utf-8",
            )

            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={
                    "layer.weight": DummyTensor(),
                    "layer.scales": DummyTensor(),
                    "layer.biases": DummyTensor(),
                },
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Checkpoint already contains MLX quantized weights",
                ):
                    model.from_pretrained(tmpdir, quantize=True, bits=4)
