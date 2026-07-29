import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx

from src.mlx_transformers.benchmark import (
    main,
    read_json,
    render_log,
    validate_result,
    validate_scenario,
)
from src.mlx_transformers.quantization import (
    QuantizationConfig,
    QuantizationInfo,
)


class FakeTokenizer:
    bos_token_id = 1

    def __call__(self, prompt, *, add_special_tokens):
        del prompt, add_special_tokens
        return {"input_ids": [4, 5, 6]}


class FakeModel:
    def generate(self, inputs, *, max_new_tokens, temp, eos_token_id):
        self.inputs = inputs
        self.settings = (max_new_tokens, temp, eos_token_id)
        for token_id in range(max_new_tokens):
            yield mx.array([token_id % 8])


class TestBenchmarkProtocol(unittest.TestCase):
    def test_builtin_scenarios_define_requested_workloads(self):
        short = read_json("short-decode-128")
        medium = read_json("prefill-512-decode-64")

        validate_scenario(short)
        validate_scenario(medium)
        self.assertEqual((short["prompt_tokens"], short["max_new_tokens"]), (64, 128))
        self.assertEqual(
            (medium["prompt_tokens"], medium["max_new_tokens"]),
            (512, 64),
        )

    def test_run_validate_and_render_without_network(self):
        scenario = {
            "protocol_version": "1.0",
            "id": "test-scenario",
            "description": "Small offline control.",
            "prompt": "hello",
            "prompt_tokens": 8,
            "max_new_tokens": 2,
            "temperature": 0.0,
            "seed": 0,
            "warmup_runs": 1,
            "measured_runs": 5,
        }
        info = QuantizationInfo.from_config(
            QuantizationConfig(group_size=32, bits=4),
            source="checkpoint",
        )
        loaded = SimpleNamespace(
            model=FakeModel(),
            tokenizer=FakeTokenizer(),
            quantization=info,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir, "scenario.json")
            result_path = Path(tmpdir, "result.json")
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
            with patch(
                "src.mlx_transformers.benchmark.load_causal_model",
                return_value=loaded,
            ):
                exit_code = main(
                    [
                        "run",
                        "--scenario",
                        str(scenario_path),
                        "--model",
                        "org/model",
                        "--revision",
                        "a" * 40,
                        "--implementation-revision",
                        "b" * 40,
                        "--local-files-only",
                        "--output",
                        str(result_path),
                    ]
                )

            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(len(result["runs"]), 5)
            self.assertEqual(loaded.model.inputs["input_ids"].shape, (1, 8))
            self.assertEqual(loaded.model.settings, (2, 0.0, ()))
            self.assertEqual(result["model"]["quantization"]["source"], "checkpoint")
            validate_result(result)

            log = render_log([str(result_path)])
            self.assertIn("test-scenario", log)
            self.assertIn("org/model", log)
            self.assertNotIn("reference-results", log)

    def test_result_requires_immutable_revision(self):
        result = self._valid_result()
        result["model"]["revision"] = "main"

        with self.assertRaisesRegex(ValueError, "40-character"):
            validate_result(result)

    def test_result_rejects_private_paths(self):
        result = self._valid_result()
        result["environment"]["note"] = "/Users/example/private-cache"

        with self.assertRaisesRegex(ValueError, "private path"):
            validate_result(result)

    def test_result_rejects_nondeterministic_tokens(self):
        result = self._valid_result()
        result["runs"][0]["token_sha256"] = "d" * 64

        with self.assertRaisesRegex(ValueError, "checksums differ"):
            validate_result(result)

    @staticmethod
    def _valid_result():
        scenario = {
            "protocol_version": "1.0",
            "id": "control",
            "description": "Control result.",
            "prompt": "hello",
            "prompt_tokens": 4,
            "max_new_tokens": 2,
            "temperature": 0.0,
            "seed": 0,
            "warmup_runs": 1,
            "measured_runs": 5,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            scenario_path = Path(tmpdir, "scenario.json")
            scenario_path.write_text(json.dumps(scenario), encoding="utf-8")
            loaded_scenario = read_json(str(scenario_path))
        result = {
            "schema_version": "1.0",
            "created_at": "2026-07-28T00:00:00+00:00",
            "invocation": [
                "mlx-transformers-benchmark",
                "run",
                "--output",
                "<result.json>",
            ],
            "scenario": {
                "definition": loaded_scenario,
                "sha256": hashlib.sha256(
                    json.dumps(
                        loaded_scenario,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest(),
            },
            "model": {
                "id": "org/model",
                "tokenizer_id": "org/model",
                "revision": "a" * 40,
                "dtype": "float32",
                "local_files_only": True,
                "quantization": None,
            },
            "environment": {
                "hardware": {
                    "architecture": "arm64",
                    "chip": "Apple M2 Max",
                    "machine_model": "Mac14,6",
                    "memory_bytes": 32 * 1024**3,
                },
                "software": {
                    "macos": "26.0",
                    "python": "3.12.9",
                    "mlx": "0.32.0",
                    "transformers": "5.14.1",
                    "mlx_transformers": "0.3.0",
                    "mlx_transformers_commit": None,
                },
                "model_active_memory_bytes": 1,
            },
            "runs": [],
            "summary": {},
        }
        run = {
            "generated_tokens": 2,
            "token_sha256": "c" * 64,
            "time_to_first_token_seconds": 0.1,
            "prefill_tokens_per_second": 40.0,
            "decode_tokens_per_second": 10.0,
            "total_seconds": 0.2,
            "peak_memory_bytes": 1024,
        }
        result["runs"] = [copy.deepcopy(run) for _ in range(5)]
        result["summary"] = {
            metric: {
                "mean": value,
                "median": value,
                "standard_deviation": 0.0,
                "minimum": value,
                "maximum": value,
            }
            for metric, value in run.items()
            if metric not in {"generated_tokens", "token_sha256"}
        }
        return result
