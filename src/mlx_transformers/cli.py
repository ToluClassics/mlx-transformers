import argparse
import json
import sys
from typing import List, Optional

import mlx.core as mx

from .inference import generate_text, load_causal_model
from .quantization import QuantizationConfig


_DTYPES = {
    "float16": mx.float16,
    "bfloat16": mx.bfloat16,
    "float32": mx.float32,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlx-transformers-generate",
        description=(
            "Run bounded text generation with a supported MLX checkpoint. "
            "Pre-quantized MLX checkpoints are detected automatically."
        ),
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Hugging Face model ID or local checkpoint directory.",
    )
    parser.add_argument("--prompt", required=True, help="Prompt to generate from.")
    parser.add_argument(
        "--tokenizer",
        help="Optional tokenizer ID/path; required for OpenELM.",
    )
    parser.add_argument("--revision", default="main", help="Hub revision to load.")
    parser.add_argument("--cache-dir", help="Hugging Face cache directory.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only a local directory or an already-cached Hub snapshot.",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(_DTYPES),
        default="float32",
        help=(
            "Floating dtype used while loading a regular checkpoint. "
            "Pre-quantized checkpoint tensors keep their stored dtypes."
        ),
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize a regular checkpoint in memory after loading it.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        help="Quantization group size; the selected MLX mode default is used if omitted.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        help="Bits per weight; the selected MLX mode default is used if omitted.",
    )
    parser.add_argument(
        "--mode",
        choices=["affine", "mxfp4", "mxfp8", "nvfp4"],
        default="affine",
        help="MLX quantization mode.",
    )
    parser.add_argument(
        "--quantize-input",
        action="store_true",
        help="Quantize Linear inputs; supported only by mxfp8 and nvfp4.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Finite maximum number of generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature; zero selects greedy decoding.",
    )
    parser.add_argument("--seed", type=int, default=0, help="MLX PRNG seed.")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Do not apply the tokenizer's chat template.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be non-negative.")
    if args.temperature < 0:
        parser.error("--temperature must be non-negative.")
    has_quantization_options = (
        args.group_size is not None
        or args.bits is not None
        or args.mode != "affine"
        or args.quantize_input
    )
    if has_quantization_options and not args.quantize:
        parser.error(
            "--group-size, --bits, --mode, and --quantize-input require --quantize. "
            "Omit them when loading an already-quantized MLX checkpoint."
        )

    quantization = None
    if args.quantize:
        try:
            quantization = QuantizationConfig(
                group_size=args.group_size,
                bits=args.bits,
                mode=args.mode,
                quantize_input=args.quantize_input,
            )
        except ValueError as error:
            parser.error(str(error))

    mx.random.seed(args.seed)
    loaded = load_causal_model(
        args.model,
        tokenizer_name_or_path=args.tokenizer,
        cache_dir=args.cache_dir,
        revision=args.revision,
        local_files_only=args.local_files_only,
        dtype=_DTYPES[args.dtype],
        quantization=quantization,
    )
    status = (
        {"source": "none"}
        if loaded.quantization is None
        else loaded.quantization.as_dict()
    )
    print(f"quantization={json.dumps(status, sort_keys=True)}", file=sys.stderr)

    result = generate_text(
        loaded.model,
        loaded.tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        use_chat_template=not args.raw_prompt,
    )
    print(result.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
