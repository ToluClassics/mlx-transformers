import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from transformers import LlamaConfig

from src.mlx_transformers.cli import main
from src.mlx_transformers.inference import (
    GenerationResult,
    LoadedCausalModel,
    load_causal_model,
    resolve_causal_model_class,
)
from src.mlx_transformers.models import LlamaForCausalLM
from src.mlx_transformers.quantization import (
    QuantizationConfig,
    QuantizationInfo,
)


def tiny_llama_config() -> LlamaConfig:
    return LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        tie_word_embeddings=False,
    )


def save_model(model, config, directory: str, *, quantization=None) -> None:
    mx.eval(model.parameters())
    mx.save_safetensors(
        str(Path(directory, "model.safetensors")),
        dict(tree_flatten(model.parameters())),
    )
    config_data = config.to_dict()
    if quantization is not None:
        config_data["quantization"] = quantization
    Path(directory, "config.json").write_text(
        json.dumps(config_data),
        encoding="utf-8",
    )


class TestQuantizedCheckpointExecution(unittest.TestCase):
    def assert_model_executes(self, model) -> None:
        inputs = {"input_ids": mx.array([[1, 5, 6]], dtype=mx.int32)}
        output = model(**inputs)
        mx.eval(output.logits)
        self.assertTrue(bool(mx.all(mx.isfinite(output.logits)).item()))

        tokens = list(
            model.generate(
                inputs,
                max_new_tokens=2,
                temp=0.0,
            )
        )
        mx.eval(tokens)
        self.assertGreaterEqual(len(tokens), 1)
        self.assertLessEqual(len(tokens), 2)

    def test_regular_checkpoint_can_be_quantized_and_generated(self):
        config = tiny_llama_config()
        source = LlamaForCausalLM(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(source, config, tmpdir)
            loaded = LlamaForCausalLM(config)
            loaded.from_pretrained(
                tmpdir,
                quantization=QuantizationConfig(group_size=32, bits=4),
            )

        parameter_keys = dict(tree_flatten(loaded.parameters()))
        self.assertTrue(any(key.endswith(".scales") for key in parameter_keys))
        self.assertEqual(loaded.quantization_info.source, "runtime")
        self.assert_model_executes(loaded)

    def test_prequantized_checkpoint_can_be_loaded_and_generated(self):
        config = tiny_llama_config()
        source = LlamaForCausalLM(config)
        nn.quantize(source, group_size=32, bits=4)

        with tempfile.TemporaryDirectory() as tmpdir:
            save_model(
                source,
                config,
                tmpdir,
                quantization={"group_size": 32, "bits": 4},
            )
            loaded = LlamaForCausalLM(config)
            loaded.from_pretrained(tmpdir)

        self.assertEqual(loaded.quantization_info.source, "checkpoint")
        self.assertEqual(loaded.quantization_info.group_size, 32)
        self.assert_model_executes(loaded)


class TestInferenceAPIAndCLI(unittest.TestCase):
    def test_resolver_uses_architecture_then_model_type(self):
        by_architecture = resolve_causal_model_class(
            SimpleNamespace(
                architectures=["LlamaForCausalLM"],
                model_type="unknown",
            )
        )
        by_model_type = resolve_causal_model_class(
            SimpleNamespace(architectures=[], model_type="llama")
        )

        self.assertIs(by_architecture, LlamaForCausalLM)
        self.assertIs(by_model_type, LlamaForCausalLM)

    def test_resolver_rejects_unsupported_model(self):
        with self.assertRaisesRegex(ValueError, "Unsupported causal model"):
            resolve_causal_model_class(
                SimpleNamespace(architectures=[], model_type="bert")
            )

    def test_openelm_requires_tokenizer_before_loading_weights(self):
        config = SimpleNamespace(
            architectures=["OpenELMForCausalLM"],
            model_type="openelm",
        )
        with patch(
            "src.mlx_transformers.inference.AutoConfig.from_pretrained",
            return_value=config,
        ), self.assertRaisesRegex(ValueError, "OpenELM.*tokenizer"):
            load_causal_model("/model")

    def test_cli_reports_quantization_and_generates(self):
        info = QuantizationInfo.from_config(
            QuantizationConfig(group_size=32, bits=4),
            source="checkpoint",
        )
        loaded = LoadedCausalModel(
            model=object(),
            tokenizer=object(),
            quantization=info,
        )
        result = GenerationResult(text="generated text", token_ids=[1, 2])

        with patch(
            "src.mlx_transformers.cli.load_causal_model",
            return_value=loaded,
        ) as mock_load, patch(
            "src.mlx_transformers.cli.generate_text",
            return_value=result,
        ) as mock_generate, patch("sys.stdout") as stdout, patch(
            "sys.stderr"
        ) as stderr:
            exit_code = main(
                [
                    "--model",
                    "/model",
                    "--prompt",
                    "hello",
                    "--local-files-only",
                ]
            )

        self.assertEqual(exit_code, 0)
        self.assertTrue(mock_load.call_args.kwargs["local_files_only"])
        self.assertEqual(mock_load.call_args.kwargs["dtype"], mx.float32)
        self.assertIsNone(mock_load.call_args.kwargs["quantization"])
        mock_generate.assert_called_once()
        self.assertIn("generated text", stdout.method_calls[0].args[0])
        self.assertIn('"source": "checkpoint"', stderr.method_calls[0].args[0])

    def test_cli_requires_quantize_for_runtime_options(self):
        with self.assertRaises(SystemExit):
            main(
                [
                    "--model",
                    "/model",
                    "--prompt",
                    "hello",
                    "--bits",
                    "4",
                ]
            )
