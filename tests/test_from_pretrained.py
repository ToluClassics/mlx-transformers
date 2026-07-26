import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from huggingface_hub.errors import LocalEntryNotFoundError

import mlx.core as mx

from src.mlx_transformers.models.base import MlxPretrainedMixin
from src.mlx_transformers.quantization import QuantizationConfig


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
        self.assertEqual(model.loaded_weights, ({"layer.weight": tensor}, True))
        self.assertIsNone(model.updated_weights)
        self.assertTrue(model.eval_called)
        self.assertEqual(model.quantization_info.source, "runtime")
        self.assertEqual(model.quantization_info.group_size, 64)
        self.assertEqual(model.config.quantization["bits"], 4)
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
        self.assertTrue(model.loaded_weights[1])
        self.assertEqual(model.config.quantization, {"group_size": 64, "bits": 4})
        self.assertIsNone(quantized_weight.dtype)
        self.assertIsNone(quantized_scales.dtype)
        self.assertIsNone(quantized_biases.dtype)

        kwargs = mock_quantize.call_args.kwargs
        predicate = kwargs["class_predicate"]

        self.assertEqual(kwargs["group_size"], 64)
        self.assertEqual(kwargs["bits"], 4)
        self.assertEqual(kwargs["mode"], "affine")
        self.assertFalse(kwargs["quantize_input"])
        self.assertTrue(predicate("layer", FakeQuantizableModule()))
        self.assertFalse(predicate("other", FakeQuantizableModule()))
        self.assertFalse(predicate("layer", object()))
        self.assertEqual(model.quantization_info.source, "checkpoint")
        self.assertEqual(model.quantization_info.group_size, 64)

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

    def test_from_pretrained_accepts_typed_quantization_config(self):
        tensor = DummyTensor()
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={"layer.weight": tensor},
            ), patch(
                "src.mlx_transformers.models.base.nn.quantize",
            ) as mock_quantize:
                model.from_pretrained(
                    tmpdir,
                    quantization=QuantizationConfig(group_size=32, bits=3),
                )

        mock_quantize.assert_called_once_with(
            model,
            group_size=32,
            bits=3,
            mode="affine",
            quantize_input=False,
            class_predicate=None,
        )
        self.assertEqual(model.quantization_info.source, "runtime")
        self.assertEqual(model.config.quantization["group_size"], 32)

    def test_from_pretrained_rejects_mixed_quantization_apis(self):
        model = DummyModel()

        with self.assertRaisesRegex(ValueError, "either quantization="):
            model.from_pretrained(
                "unused",
                quantize=True,
                quantization=QuantizationConfig(),
            )


class TestFromPretrainedLoadingContract(unittest.TestCase):
    def test_from_pretrained_accepts_explicit_bfloat16_dtype(self):
        model = DummyModel()
        tensor = DummyTensor()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={"layer.weight": tensor},
            ):
                model.from_pretrained(tmpdir, dtype=mx.bfloat16)

        self.assertEqual(tensor.dtype, mx.bfloat16)

    def test_from_pretrained_rejects_conflicting_dtype_arguments(self):
        model = DummyModel()

        with self.assertRaisesRegex(ValueError, "only one of float16"):
            model.from_pretrained(
                "unused",
                float16=True,
                dtype=mx.bfloat16,
            )

    def test_from_pretrained_rejects_missing_required_tensors(self):
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={},
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "missing required model tensor key.*layer.weight",
                ):
                    model.from_pretrained(tmpdir)

        self.assertIsNone(model.loaded_weights)
        self.assertFalse(model.eval_called)

    def test_from_pretrained_rejects_unexpected_tensors(self):
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={
                    "layer.weight": DummyTensor(),
                    "other.weight": DummyTensor(),
                },
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "do not map to this model.*other.weight",
                ):
                    model.from_pretrained(tmpdir)

    def test_from_pretrained_ignores_known_transformers_buffers(self):
        model = DummyModel()
        tensor = DummyTensor()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={
                    "layer.weight": tensor,
                    "embeddings.position_ids": DummyTensor(),
                    "bert.pooler.dense.bias": DummyTensor(),
                    "bert.pooler.dense.weight": DummyTensor(),
                },
            ):
                model.from_pretrained(tmpdir)

        self.assertEqual(model.loaded_weights, ({"layer.weight": tensor}, True))

    def test_from_pretrained_keeps_parameters_derived_from_config(self):
        model = DummyModel()
        tensor = DummyTensor()
        derived_tensor = object()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={"layer.weight": tensor},
            ), patch(
                "src.mlx_transformers.models.base.tree_flatten",
                return_value=[
                    ("layer.weight", object()),
                    ("layer.rotary_emb.inv_freq", derived_tensor),
                ],
            ):
                model.from_pretrained(tmpdir)

        self.assertEqual(
            model.loaded_weights,
            (
                {
                    "layer.weight": tensor,
                    "layer.rotary_emb.inv_freq": derived_tensor,
                },
                True,
            ),
        )

    def test_from_pretrained_rejects_duplicate_shard_keys(self):
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model-00001-of-00002.safetensors").touch()
            Path(tmpdir, "model-00002-of-00002.safetensors").touch()
            with patch(
                "src.mlx_transformers.models.base.mx.load",
                side_effect=[
                    {"layer.weight": DummyTensor()},
                    {"layer.weight": DummyTensor()},
                ],
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Duplicate tensor key.*layer.weight",
                ):
                    model.from_pretrained(tmpdir)

    def test_from_pretrained_uses_safetensors_index_shards(self):
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            shard = Path(tmpdir, "model-00001-of-00001.safetensors")
            shard.touch()
            Path(tmpdir, "unreferenced.safetensors").touch()
            Path(tmpdir, "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.weight": shard.name}}),
                encoding="utf-8",
            )
            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={"layer.weight": DummyTensor()},
            ) as mock_load:
                model.from_pretrained(tmpdir)

        mock_load.assert_called_once_with(str(shard))

    def test_from_pretrained_rejects_bin_only_checkpoint(self):
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "pytorch_model.bin").touch()
            with self.assertRaisesRegex(ValueError, "supports safetensors only"):
                model.from_pretrained(tmpdir)

    def test_from_pretrained_forwards_hub_resolution_controls(self):
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            with patch(
                "src.mlx_transformers.models.base.snapshot_download",
                return_value=tmpdir,
            ) as mock_snapshot, patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={"layer.weight": DummyTensor()},
            ):
                model.from_pretrained(
                    "org/model",
                    cache_dir="/tmp/test-cache",
                    revision="revision-sha",
                    local_files_only=True,
                    token="test-token",
                    max_workers=2,
                )

        mock_snapshot.assert_called_once_with(
            repo_id="org/model",
            allow_patterns=MlxPretrainedMixin._CHECKPOINT_PATTERNS,
            cache_dir="/tmp/test-cache",
            local_files_only=True,
            max_workers=2,
            revision="revision-sha",
            token="test-token",
        )

    def test_from_pretrained_reports_offline_cache_miss(self):
        model = DummyModel()

        with patch(
            "src.mlx_transformers.models.base.snapshot_download",
            side_effect=LocalEntryNotFoundError("not cached"),
        ):
            with self.assertRaisesRegex(
                FileNotFoundError,
                "No cached snapshot.*Disable local_files_only",
            ):
                model.from_pretrained("org/model", local_files_only=True)

    def test_from_pretrained_warns_that_remote_code_is_not_executed(self):
        model = DummyModel()

        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "model.safetensors").touch()
            with patch(
                "src.mlx_transformers.models.base.mx.load",
                return_value={"layer.weight": DummyTensor()},
            ), patch(
                "src.mlx_transformers.models.base.warnings.warn",
            ) as mock_warn:
                model.from_pretrained(tmpdir, trust_remote_code=True)

        self.assertIn("trust_remote_code has no effect", mock_warn.call_args.args[0])
