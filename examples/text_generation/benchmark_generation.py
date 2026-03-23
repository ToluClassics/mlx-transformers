import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
from transformers import AutoConfig, AutoTokenizer

from common import get_eos_token_ids
from mlx_transformers.models import (
    Gemma3ForCausalLM,
    LlamaForCausalLM,
    OpenELMForCausalLM,
    PersimmonForCausalLM,
    Phi3ForCausalLM,
    PhiForCausalLM,
    Qwen3ForCausalLM,
)

ARCHITECTURE_2_CLASS = {
    "Gemma3ForCausalLM": Gemma3ForCausalLM,
    "LlamaForCausalLM": LlamaForCausalLM,
    "OpenELMForCausalLM": OpenELMForCausalLM,
    "PersimmonForCausalLM": PersimmonForCausalLM,
    "Phi3ForCausalLM": Phi3ForCausalLM,
    "PhiForCausalLM": PhiForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}

MODEL_TYPE_2_CLASS = {
    "gemma3_text": Gemma3ForCausalLM,
    "llama": LlamaForCausalLM,
    "openelm": OpenELMForCausalLM,
    "persimmon": PersimmonForCausalLM,
    "phi": PhiForCausalLM,
    "phi3": Phi3ForCausalLM,
    "qwen3": Qwen3ForCausalLM,
}


@dataclass
class BenchmarkResult:
    label: str
    model_name: str
    bucket: str
    samples: int
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    prefill_tokens_per_second: float
    decode_seconds: float
    decode_tokens_per_second: float
    full_seconds: float


@dataclass
class PromptExample:
    bucket: str
    prompt_tokens: int
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None


def parse_model_spec(spec: str) -> Tuple[str, str]:
    if "=" in spec:
        label, model_name = spec.split("=", 1)
        return label.strip(), model_name.strip()
    return spec.strip(), spec.strip()


def resolve_model_class(config) -> type[Any]:
    architectures = getattr(config, "architectures", []) or []
    for architecture in architectures:
        if architecture in ARCHITECTURE_2_CLASS:
            return ARCHITECTURE_2_CLASS[architecture]

    model_type = getattr(config, "model_type", None)
    if model_type in MODEL_TYPE_2_CLASS:
        return MODEL_TYPE_2_CLASS[model_type]

    raise ValueError(
        "Unsupported benchmark model architecture. "
        f"architectures={architectures!r}, model_type={model_type!r}. "
        "Supported text model families are: gemma3_text, llama, openelm, "
        "persimmon, phi, phi3, qwen3."
    )


def resolve_tokenizer_name(model_name: str, config) -> str:
    if getattr(config, "model_type", None) == "openelm":
        return "meta-llama/Llama-2-7b-hf"
    return model_name


def coerce_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    text_chunks.append(str(text))
            elif isinstance(item, str):
                text_chunks.append(item)
        return "\n".join(text_chunks)
    return str(content)


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized_messages = []
    for message in messages:
        role = message.get("role")
        content = coerce_message_content(message.get("content", ""))
        if not role or not content:
            continue
        normalized_messages.append({"role": role, "content": content})

    if normalized_messages and normalized_messages[-1]["role"] == "assistant":
        normalized_messages = normalized_messages[:-1]

    return normalized_messages


def fallback_prompt_from_messages(messages: List[Dict[str, str]]) -> str:
    rendered_messages = [
        f"{message['role']}: {message['content']}" for message in messages
    ]
    rendered_messages.append("assistant:")
    return "\n\n".join(rendered_messages)


def build_inputs(
    tokenizer,
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, mx.array]:
    if messages is None:
        messages = [{"role": "user", "content": prompt or ""}]

    if getattr(tokenizer, "chat_template", None):
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="np",
            )
            return {key: mx.array(value) for key, value in inputs.items()}
        except (AttributeError, TypeError, ValueError):
            pass

    if prompt is None:
        prompt = fallback_prompt_from_messages(messages)

    inputs = tokenizer(prompt, return_tensors="np", truncation=True)
    return {key: mx.array(value) for key, value in inputs.items()}


def parse_bucket_spec(spec: str) -> Tuple[int, int]:
    lower_bound, upper_bound = spec.split(":", 1)
    return int(lower_bound), int(upper_bound)


def format_bucket_label(lower_bound: int, upper_bound: int) -> str:
    if upper_bound >= sys.maxsize:
        return f"{lower_bound}+"
    return f"{lower_bound}-{upper_bound}"


def find_bucket(
    prompt_tokens: int,
    bucket_specs: List[Tuple[int, int]],
) -> Optional[str]:
    for lower_bound, upper_bound in bucket_specs:
        if lower_bound <= prompt_tokens <= upper_bound:
            return format_bucket_label(lower_bound, upper_bound)
    return None


def load_ultrachat_examples(tokenizer, args) -> List[PromptExample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "UltraChat benchmarking requires the 'datasets' package. "
            "Install it with `pip install datasets`."
        ) from exc

    bucket_specs = [parse_bucket_spec(spec) for spec in args.bucket]
    selected_examples = {
        format_bucket_label(lower_bound, upper_bound): []
        for lower_bound, upper_bound in bucket_specs
    }

    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split=args.dataset_split,
    ).shuffle(seed=args.seed)

    for row_index, row in enumerate(dataset):
        if row_index >= args.dataset_max_samples and all(
            len(examples) >= args.samples_per_bucket
            for examples in selected_examples.values()
        ):
            break

        messages = normalize_messages(row.get("messages", []))
        if not messages:
            continue

        inputs = build_inputs(tokenizer, messages=messages)
        prompt_tokens = int(inputs["input_ids"].shape[-1])
        bucket = find_bucket(prompt_tokens, bucket_specs)
        if bucket is None:
            continue

        if len(selected_examples[bucket]) >= args.samples_per_bucket:
            continue

        selected_examples[bucket].append(
            PromptExample(
                bucket=bucket,
                prompt_tokens=prompt_tokens,
                messages=messages,
            )
        )

        if all(
            len(examples) >= args.samples_per_bucket
            for examples in selected_examples.values()
        ):
            break

    missing_buckets = [
        bucket
        for bucket, examples in selected_examples.items()
        if len(examples) < args.samples_per_bucket
    ]
    if missing_buckets:
        raise ValueError(
            "Unable to sample enough UltraChat prompts for buckets: "
            + ", ".join(missing_buckets)
        )

    ordered_examples = []
    for lower_bound, upper_bound in bucket_specs:
        bucket = format_bucket_label(lower_bound, upper_bound)
        ordered_examples.extend(selected_examples[bucket])

    return ordered_examples


def load_prompt_examples(tokenizer, args) -> List[PromptExample]:
    if args.dataset == "ultrachat":
        return load_ultrachat_examples(tokenizer, args)

    return [
        PromptExample(
            bucket="single",
            prompt_tokens=0,
            prompt=args.prompt,
        )
    ]


def load_model_and_tokenizer(model_name: str, args):
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_class = resolve_model_class(config)
    model = model_class(config)
    model.from_pretrained(
        model_name,
        trust_remote_code=True,
        quantize=args.quantize,
        group_size=args.group_size,
        bits=args.bits,
        mode=args.mode,
        quantize_input=args.quantize_input,
    )

    tokenizer_name = resolve_tokenizer_name(model_name, config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return model, tokenizer


def benchmark_once(
    model,
    tokenizer,
    model_name: str,
    example: PromptExample,
    args,
) -> BenchmarkResult:
    inputs = build_inputs(
        tokenizer,
        prompt=example.prompt,
        messages=example.messages,
    )
    prompt_tokens = int(inputs["input_ids"].shape[-1])
    eos_token_ids = get_eos_token_ids(model_name, tokenizer)

    first_token_time = None
    generated_tokens = 0
    start = time.time()

    for token in model.generate(inputs, max_length=args.max_tokens, temp=args.temp):
        mx.eval(token)
        token_id = int(token.item())
        now = time.time()

        if first_token_time is None:
            first_token_time = now

        if token_id in eos_token_ids:
            break

        generated_tokens += 1

    end = time.time()
    if first_token_time is None:
        first_token_time = end

    prefill_seconds = first_token_time - start
    decode_seconds = max(end - first_token_time, 0.0)
    full_seconds = end - start
    decode_tokens = max(generated_tokens - 1, 0)

    return BenchmarkResult(
        label="",
        model_name=model_name,
        bucket=example.bucket,
        samples=1,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        prefill_seconds=prefill_seconds,
        prefill_tokens_per_second=(
            prompt_tokens / prefill_seconds if prefill_seconds > 0 else 0.0
        ),
        decode_seconds=decode_seconds,
        decode_tokens_per_second=(
            decode_tokens / decode_seconds
            if decode_seconds > 0 and decode_tokens > 0
            else 0.0
        ),
        full_seconds=full_seconds,
    )


def average_results(
    label: str,
    model_name: str,
    bucket: str,
    results: List[BenchmarkResult],
) -> BenchmarkResult:
    count = len(results)
    return BenchmarkResult(
        label=label,
        model_name=model_name,
        bucket=bucket,
        samples=count,
        prompt_tokens=round(sum(r.prompt_tokens for r in results) / count),
        generated_tokens=round(sum(r.generated_tokens for r in results) / count),
        prefill_seconds=sum(r.prefill_seconds for r in results) / count,
        prefill_tokens_per_second=sum(r.prefill_tokens_per_second for r in results)
        / count,
        decode_seconds=sum(r.decode_seconds for r in results) / count,
        decode_tokens_per_second=sum(r.decode_tokens_per_second for r in results)
        / count,
        full_seconds=sum(r.full_seconds for r in results) / count,
    )


def format_markdown(results: List[BenchmarkResult]) -> str:
    lines = [
        (
            "| Label | Hugging Face model | Bucket | Samples | Prompt tokens | "
            "New tokens | Prefill (s) | Prefill tok/s | Decode (s) | "
            "Decode tok/s | Full (s) |"
        ),
        (
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | "
            "---: | ---: | ---: |"
        ),
    ]

    for result in results:
        lines.append(
            "| "
            f"{result.label} | "
            f"{result.model_name} | "
            f"{result.bucket} | "
            f"{result.samples} | "
            f"{result.prompt_tokens} | "
            f"{result.generated_tokens} | "
            f"{result.prefill_seconds:.3f} | "
            f"{result.prefill_tokens_per_second:.2f} | "
            f"{result.decode_seconds:.3f} | "
            f"{result.decode_tokens_per_second:.2f} | "
            f"{result.full_seconds:.3f} |"
        )

    return "\n".join(lines)


def format_json(results: List[BenchmarkResult]) -> str:
    payload = [asdict(result) for result in results]
    return json.dumps(payload, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Benchmark text generation models")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Benchmark target as 'label=repo/name' or just 'repo/name'. "
            "Repeat for multiple models."
        ),
    )
    parser.add_argument(
        "--prompt",
        default="Explain grouped-query attention in one paragraph.",
        help="Prompt used when --dataset is not set.",
    )
    parser.add_argument(
        "--dataset",
        choices=("ultrachat",),
        default=None,
        help="Optional prompt dataset for multi-size benchmarking.",
    )
    parser.add_argument(
        "--dataset-split",
        default="train_sft",
        help="Dataset split to load when --dataset is set.",
    )
    parser.add_argument(
        "--bucket",
        action="append",
        default=None,
        help=(
            "Prompt-token bucket as min:max. Repeat to benchmark multiple "
            "prompt sizes."
        ),
    )
    parser.add_argument(
        "--samples-per-bucket",
        type=int,
        default=10,
        help="How many prompts to sample from each dataset bucket.",
    )
    parser.add_argument(
        "--dataset-max-samples",
        type=int,
        default=100,
        help="Maximum dataset rows to scan while filling prompt buckets.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Maximum number of generated tokens to benchmark.",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measured runs per model.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup runs per model before measurement.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to write the formatted benchmark output.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize the checkpoint after loading.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=None,
        help="Quantization group size passed to mlx.nn.quantize.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="Number of bits per quantized weight.",
    )
    parser.add_argument(
        "--mode",
        default="affine",
        help="Quantization mode passed to mlx.nn.quantize.",
    )
    parser.add_argument(
        "--quantize-input",
        action="store_true",
        help="Quantize supported layer inputs; only valid with mode nvfp4 or mxfp8.",
    )
    args = parser.parse_args()

    mx.random.seed(args.seed)
    mx.set_default_device(mx.gpu)

    if args.bucket is None:
        args.bucket = ["1:128", "129:512", "513:1024", "1025:2048"]

    averaged_results = []
    prompt_examples = None

    for model_spec in args.model:
        label, model_name = parse_model_spec(model_spec)
        print(f"[INFO] Loading {label} ({model_name})", file=sys.stderr)
        model, tokenizer = load_model_and_tokenizer(model_name, args)

        if prompt_examples is None:
            prompt_examples = load_prompt_examples(tokenizer, args)

        bucketed_examples: Dict[str, List[PromptExample]] = {}
        for example in prompt_examples:
            bucketed_examples.setdefault(example.bucket, []).append(example)

        for bucket, examples in bucketed_examples.items():
            for _ in range(args.warmup_runs):
                benchmark_once(
                    model,
                    tokenizer,
                    model_name,
                    examples[0],
                    args,
                )

            measured_results = []
            for run_idx in range(args.runs):
                print(
                    f"[INFO] Benchmarking {label} bucket {bucket} "
                    f"run {run_idx + 1}/{args.runs}",
                    file=sys.stderr,
                )
                for example in examples:
                    result = benchmark_once(
                        model,
                        tokenizer,
                        model_name,
                        example,
                        args,
                    )
                    measured_results.append(result)

            averaged_result = average_results(
                label,
                model_name,
                bucket,
                measured_results,
            )
            averaged_results.append(averaged_result)

    if args.format == "json":
        output = format_json(averaged_results)
    else:
        output = format_markdown(averaged_results)

    if args.output_file:
        with open(args.output_file, "w", encoding="utf-8") as output_file:
            output_file.write(output)
            output_file.write("\n")

    print(output)


if __name__ == "__main__":
    main()
