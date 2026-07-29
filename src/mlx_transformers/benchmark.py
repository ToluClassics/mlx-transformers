import argparse
import hashlib
import json
import math
import platform
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, distribution, version
from importlib.resources import files
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlx.core as mx

from .inference import load_causal_model


SCHEMA_VERSION = "1.0"
_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}
_RUN_METRICS = (
    "time_to_first_token_seconds",
    "prefill_tokens_per_second",
    "decode_tokens_per_second",
    "total_seconds",
    "peak_memory_bytes",
)


def read_json(path_or_name: str) -> Dict[str, Any]:
    path = Path(path_or_name)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))

    filename = path_or_name
    if not filename.endswith(".json"):
        filename += ".json"
    resource = files(__package__).joinpath("benchmark_data", filename)
    if not resource.is_file():
        raise ValueError(f"Unknown benchmark scenario or JSON file: {path_or_name}")
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_scenario(scenario: Dict[str, Any]) -> None:
    required = {
        "protocol_version",
        "id",
        "description",
        "prompt",
        "prompt_tokens",
        "max_new_tokens",
        "temperature",
        "seed",
        "warmup_runs",
        "measured_runs",
    }
    missing = sorted(required - set(scenario))
    if missing:
        raise ValueError(f"Scenario is missing required keys: {', '.join(missing)}")
    if scenario["protocol_version"] != SCHEMA_VERSION:
        raise ValueError("Only benchmark protocol_version '1.0' is supported.")
    if not isinstance(scenario["id"], str) or not scenario["id"]:
        raise ValueError("Scenario id must be a non-empty string.")
    if not isinstance(scenario["prompt"], str) or not scenario["prompt"].strip():
        raise ValueError("Scenario prompt must be a non-empty string.")
    for key in ("prompt_tokens", "max_new_tokens", "warmup_runs", "measured_runs"):
        if not isinstance(scenario[key], int) or isinstance(scenario[key], bool):
            raise ValueError(f"Scenario {key} must be an integer.")
    if scenario["prompt_tokens"] < 1 or scenario["max_new_tokens"] < 2:
        raise ValueError("Scenario token counts must be positive.")
    if scenario["warmup_runs"] < 1 or scenario["measured_runs"] < 5:
        raise ValueError("Protocol v1 requires at least one warmup and five runs.")
    if scenario["temperature"] != 0.0:
        raise ValueError("Protocol v1 requires deterministic temperature 0.0.")
    if not isinstance(scenario["seed"], int) or isinstance(scenario["seed"], bool):
        raise ValueError("Scenario seed must be an integer.")


def validate_result(result: Dict[str, Any]) -> None:
    required = {
        "schema_version",
        "created_at",
        "invocation",
        "scenario",
        "model",
        "environment",
        "runs",
        "summary",
    }
    missing = sorted(required - set(result))
    if missing:
        raise ValueError(f"Result is missing required keys: {', '.join(missing)}")
    if result["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Only benchmark result schema_version '1.0' is supported.")
    try:
        datetime.fromisoformat(result["created_at"])
    except (TypeError, ValueError) as error:
        raise ValueError("Result created_at must be an ISO 8601 timestamp.") from error
    invocation = result["invocation"]
    if not invocation or not all(isinstance(value, str) for value in invocation):
        raise ValueError("Result invocation must be a non-empty string list.")

    scenario = result["scenario"]["definition"]
    validate_scenario(scenario)
    expected_hash = hashlib.sha256(
        json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if result["scenario"].get("sha256") != expected_hash:
        raise ValueError("Scenario checksum does not match its embedded definition.")

    model = result["model"]
    if not _MODEL_ID_PATTERN.fullmatch(model.get("id", "")):
        raise ValueError("Result model id must be a public Hub owner/name.")
    if not _MODEL_ID_PATTERN.fullmatch(model.get("tokenizer_id", "")):
        raise ValueError("Result tokenizer id must be a public Hub owner/name.")
    if not _REVISION_PATTERN.fullmatch(model.get("revision", "")):
        raise ValueError("Result model revision must be a 40-character commit.")
    if model.get("dtype") not in _DTYPES:
        raise ValueError("Result model dtype is invalid.")
    if not isinstance(model.get("local_files_only"), bool):
        raise ValueError("Result local_files_only must be a boolean.")

    environment = result["environment"]
    hardware = environment.get("hardware", {})
    software = environment.get("software", {})
    for key in ("architecture", "chip", "machine_model", "memory_bytes"):
        if key not in hardware:
            raise ValueError(f"Result hardware is missing {key}.")
    for key in ("macos", "python", "mlx", "transformers", "mlx_transformers"):
        if key not in software:
            raise ValueError(f"Result software is missing {key}.")
    implementation_revision = software.get("mlx_transformers_commit")
    if implementation_revision is not None and not _REVISION_PATTERN.fullmatch(
        implementation_revision
    ):
        raise ValueError("Result implementation revision must be a commit or null.")

    runs = result["runs"]
    if len(runs) != scenario["measured_runs"]:
        raise ValueError("Result run count does not match the scenario.")
    for run_index, run in enumerate(runs, start=1):
        if run.get("generated_tokens") != scenario["max_new_tokens"]:
            raise ValueError(f"Run {run_index} did not generate the required tokens.")
        if not re.fullmatch(r"[0-9a-f]{64}", run.get("token_sha256", "")):
            raise ValueError(f"Run {run_index} has an invalid token checksum.")
        for metric in _RUN_METRICS:
            value = run.get(metric)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Run {run_index} has invalid metric {metric}.")
    if len({run["token_sha256"] for run in runs}) != 1:
        raise ValueError("Greedy token checksums differ between measured runs.")
    expected_summary = summarize_runs(runs)
    for metric in _RUN_METRICS:
        statistics_for_metric = result["summary"].get(metric, {})
        for statistic in (
            "mean",
            "median",
            "standard_deviation",
            "minimum",
            "maximum",
        ):
            value = statistics_for_metric.get(statistic)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Result summary has invalid {metric}.{statistic}.")
            expected = expected_summary[metric][statistic]
            if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Result summary does not match runs for {metric}.")

    serialized = json.dumps(result)
    if any(marker in serialized for marker in ("/Users/", "/home/", "HF_TOKEN")):
        raise ValueError("Result contains a private path or credential marker.")


def build_fixed_inputs(tokenizer: Any, scenario: Dict[str, Any]) -> Dict[str, mx.array]:
    encoded = tokenizer(scenario["prompt"], add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]
    if not token_ids:
        raise ValueError("The scenario prompt produced no tokens.")

    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if bos_token_id is not None:
        token_ids = [bos_token_id, *token_ids]
    target = scenario["prompt_tokens"]
    repeated = (token_ids * ((target + len(token_ids) - 1) // len(token_ids)))[:target]
    input_ids = mx.array([repeated], dtype=mx.int32)
    return {
        "input_ids": input_ids,
        "attention_mask": mx.ones_like(input_ids),
    }


def benchmark_once(
    model: Any,
    inputs: Dict[str, mx.array],
    max_new_tokens: int,
) -> Dict[str, Any]:
    mx.clear_cache()
    mx.reset_peak_memory()
    started_at = time.perf_counter()
    tokens = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        temp=0.0,
        eos_token_id=(),
    )

    first_token = next(tokens)
    mx.eval(first_token)
    token_hash = hashlib.sha256()
    token_hash.update(int(first_token.item()).to_bytes(8, "little", signed=True))
    first_token_at = time.perf_counter()
    generated_tokens = 1
    for token in tokens:
        mx.eval(token)
        token_hash.update(int(token.item()).to_bytes(8, "little", signed=True))
        generated_tokens += 1
    completed_at = time.perf_counter()

    if generated_tokens != max_new_tokens:
        raise RuntimeError(
            f"Expected {max_new_tokens} tokens, generated {generated_tokens}."
        )
    prefill_seconds = first_token_at - started_at
    decode_seconds = completed_at - first_token_at
    return {
        "generated_tokens": generated_tokens,
        "token_sha256": token_hash.hexdigest(),
        "time_to_first_token_seconds": prefill_seconds,
        "prefill_tokens_per_second": (
            int(inputs["input_ids"].shape[-1]) / prefill_seconds
        ),
        "decode_tokens_per_second": (
            (generated_tokens - 1) / decode_seconds if decode_seconds > 0 else 0.0
        ),
        "total_seconds": completed_at - started_at,
        "peak_memory_bytes": mx.get_peak_memory(),
    }


def summarize_runs(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    for metric in _RUN_METRICS:
        values = [run[metric] for run in runs]
        summary[metric] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "standard_deviation": statistics.stdev(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    return summary


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    scenario = read_json(args.scenario)
    validate_scenario(scenario)
    if not _MODEL_ID_PATTERN.fullmatch(args.model):
        raise ValueError("--model must be a public Hugging Face owner/name.")
    if args.tokenizer and not _MODEL_ID_PATTERN.fullmatch(args.tokenizer):
        raise ValueError("--tokenizer must be a public Hugging Face owner/name.")
    if not _REVISION_PATTERN.fullmatch(args.revision):
        raise ValueError("--revision must be a 40-character Hub commit.")
    if args.implementation_revision and not _REVISION_PATTERN.fullmatch(
        args.implementation_revision
    ):
        raise ValueError("--implementation-revision must be a 40-character commit.")
    try:
        package = distribution("mlx-transformers")
        package_version = package.version
        direct_url = json.loads(package.read_text("direct_url.json") or "{}")
    except PackageNotFoundError:
        package_version = "source"
        direct_url = {}
    if direct_url.get("url", "").startswith("file:") and not (
        args.implementation_revision
    ):
        raise ValueError(
            "--implementation-revision is required for an editable/source install."
        )

    mx.random.seed(scenario["seed"])
    loaded = load_causal_model(
        args.model,
        tokenizer_name_or_path=args.tokenizer,
        cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
        dtype=_DTYPES[args.dtype],
    )
    inputs = build_fixed_inputs(loaded.tokenizer, scenario)
    for _ in range(scenario["warmup_runs"]):
        benchmark_once(loaded.model, inputs, scenario["max_new_tokens"])
    model_active_memory_bytes = mx.get_active_memory()

    runs = [
        benchmark_once(loaded.model, inputs, scenario["max_new_tokens"])
        for _ in range(scenario["measured_runs"])
    ]
    scenario_hash = hashlib.sha256(
        json.dumps(scenario, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def sysctl(name: str) -> str:
        process = subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True,
            check=False,
            text=True,
        )
        return process.stdout.strip() or "unknown"

    environment = {
        "hardware": {
            "architecture": platform.machine(),
            "chip": sysctl("machdep.cpu.brand_string"),
            "machine_model": sysctl("hw.model"),
            "memory_bytes": int(sysctl("hw.memsize")),
        },
        "software": {
            "macos": platform.mac_ver()[0],
            "python": platform.python_version(),
            "mlx": version("mlx"),
            "transformers": version("transformers"),
            "mlx_transformers": package_version,
            "mlx_transformers_commit": args.implementation_revision,
        },
        "model_active_memory_bytes": model_active_memory_bytes,
    }
    invocation = [
        "mlx-transformers-benchmark",
        "run",
        "--scenario",
        scenario["id"],
        "--model",
        args.model,
        "--revision",
        args.revision,
        "--dtype",
        args.dtype,
        "--output",
        "<result.json>",
    ]
    if args.tokenizer:
        invocation.extend(["--tokenizer", args.tokenizer])
    if args.local_files_only:
        invocation.append("--local-files-only")
    if args.implementation_revision:
        invocation.extend(["--implementation-revision", args.implementation_revision])
    result = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "invocation": invocation,
        "scenario": {
            "definition": scenario,
            "sha256": scenario_hash,
        },
        "model": {
            "id": args.model,
            "tokenizer_id": args.tokenizer or args.model,
            "revision": args.revision,
            "dtype": args.dtype,
            "local_files_only": args.local_files_only,
            "quantization": (
                loaded.quantization.as_dict()
                if loaded.quantization is not None
                else None
            ),
        },
        "environment": environment,
        "runs": runs,
        "summary": summarize_runs(runs),
    }
    validate_result(result)
    return result


def render_log(paths: List[str]) -> str:
    rows = []
    for path_string in paths:
        result = read_json(path_string)
        validate_result(result)
        scenario = result["scenario"]["definition"]
        hardware = result["environment"]["hardware"]
        quantization = result["model"]["quantization"]
        quantized = (
            "none"
            if quantization is None
            else (
                f"{quantization['mode']} {quantization['bits']}-bit/"
                f"group-{quantization['group_size']}"
            )
        )
        summary = result["summary"]
        rows.append(
            (
                scenario["id"],
                hardware["chip"],
                f"{hardware['memory_bytes'] / (1024**3):.0f} GB",
                result["model"]["id"],
                quantized,
                scenario["prompt_tokens"],
                scenario["max_new_tokens"],
                summary["time_to_first_token_seconds"]["median"] * 1000,
                summary["prefill_tokens_per_second"]["median"],
                summary["decode_tokens_per_second"]["median"],
                summary["peak_memory_bytes"]["maximum"] / (1024**3),
                len(result["runs"]),
            )
        )

    lines = [
        "# MLX Transformers Reproducibility Log",
        "",
        "Validated protocol v1 results.",
        "",
        (
            "| Scenario | Chip | Memory | Model | Quantization | Prompt | "
            "Decode | TTFT | Prefill | Decode rate | Peak | Runs |"
        ),
        (
            "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |"
        ),
    ]
    for row in sorted(rows):
        lines.append(
            "| "
            f"{row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} | "
            f"{row[5]} | {row[6]} | {row[7]:.1f} ms | "
            f"{row[8]:.1f} tok/s | {row[9]:.1f} tok/s | {row[10]:.2f} GiB | "
            f"{row[11]} |"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mlx-transformers-benchmark")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="Run one protocol v1 scenario.")
    run.add_argument("--scenario", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--revision", required=True)
    run.add_argument("--tokenizer")
    run.add_argument("--cache-dir")
    run.add_argument("--local-files-only", action="store_true")
    run.add_argument("--dtype", choices=sorted(_DTYPES), default="float32")
    run.add_argument("--implementation-revision")
    run.add_argument("--output", required=True)

    validate = commands.add_parser("validate", help="Validate result JSON.")
    validate.add_argument("results", nargs="+")

    render = commands.add_parser(
        "render",
        help="Render validated results as a reproducibility log.",
    )
    render.add_argument("results", nargs="+")
    render.add_argument("--output")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            result = run_benchmark(args)
            Path(args.output).write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(args.output)
        elif args.command == "validate":
            for path in args.results:
                validate_result(read_json(path))
                print(f"{path}: valid")
        else:
            log = render_log(args.results)
            if args.output:
                Path(args.output).write_text(log, encoding="utf-8")
                print(args.output)
            else:
                print(log, end="")
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
