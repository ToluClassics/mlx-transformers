import argparse
import time
from typing import Tuple

import mlx.core as mx
from transformers import AutoTokenizer, LlamaConfig

from mlx_transformers.models import LlamaForCausalLM as MlxLlamaForCausalLM
from common import get_eos_token_ids


def tic():
    "Return generation time in seconds"
    return time.time()


def toc(msg, start):
    "Return generation time in seconds and a message"
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def load_model(model_name: str, args) -> Tuple[MlxLlamaForCausalLM, AutoTokenizer]:
    """
    Load a quantized LLaMA model and tokenizer from Hugging Face.

    Args:
        model_name: Name of the LLaMA checkpoint to load
        args: Parsed CLI arguments controlling quantization

    Returns:
        The quantized model and tokenizer
    """
    config = LlamaConfig.from_pretrained(model_name)

    model = MlxLlamaForCausalLM(config)
    model.from_pretrained(
        model_name,
        quantize=True,
        group_size=args.group_size,
        bits=args.bits,
        mode=args.mode,
        quantize_input=args.quantize_input,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def generate(model: MlxLlamaForCausalLM, tokenizer: AutoTokenizer, args):
    print(args.prompt)
    messages = [{"role": "user", "content": args.prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="np",
    )

    inputs = {key: mx.array(v) for key, v in inputs.items()}
    eos_token_ids = get_eos_token_ids(args.model_name, tokenizer)

    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(inputs, max_length=args.max_tokens, temp=args.temp):
        mx.eval(token)
        token_id = int(token.item())

        if prompt_processing is None:
            prompt_processing = toc("Prompt processing", start)

        if token_id in eos_token_ids:
            break

        tokens.append(token)

        if (len(tokens) % args.write_every) == 0:
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    mx.eval(tokens)
    full_gen = toc("Full generation", start)
    s = tokenizer.decode([t.item() for t in tokens])
    print(s[skip:], flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantized LLaMA inference script")
    parser.add_argument(
        "--model-name",
        help="The model name to load",
        default="meta-llama/Llama-3.2-1B-Instruct",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model.",
        default="Write a short explanation of weight quantization.",
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=256, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="The sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")
    parser.add_argument(
        "--group-size",
        type=int,
        default=64,
        help="Quantization group size passed to mlx.nn.quantize",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="Number of bits per quantized weight",
    )
    parser.add_argument(
        "--mode",
        default="affine",
        help="Quantization mode, for example affine, mxfp4, nvfp4, or mxfp8",
    )
    parser.add_argument(
        "--quantize-input",
        action="store_true",
        help="Quantize supported layer inputs; requires mode nvfp4 or mxfp8",
    )

    args = parser.parse_args()

    mx.random.seed(args.seed)
    mx.set_default_device(mx.gpu)

    model, tokenizer = load_model(args.model_name, args)

    generate(model, tokenizer, args)
