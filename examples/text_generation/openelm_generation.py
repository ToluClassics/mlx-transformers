import argparse
import os
import time
from typing import Tuple

import mlx.core as mx
from transformers import AutoTokenizer, AutoConfig

from mlx_transformers.models import OpenELMForCausalLM as MlxOpenELMForCausalLM


def tic():
    "Return generation time in seconds"
    return time.time()


def toc(msg, start):
    "Return generation time in seconds and a message"
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def load_model(
    model_name: str, mlx_model_class
) -> Tuple[MlxOpenELMForCausalLM, AutoTokenizer]:
    """
    Load a llama model and tokenizer from the given model name and weights.

    Args:
        model_name (str): Name of the llama model to load
        model_weights (str): Path to the model weights
        hgf_model_class: Huggingface model class
        mlx_model_class: Mlx model class

    Returns:
        _type_: _description_
    """
    config = AutoConfig.from_pretrained(model_name)
    os.path.dirname(os.path.realpath(__file__))

    model = mlx_model_class(config)
    model.from_pretrained(
        model_name,
        huggingface_model_architecture="AutoModelForCausalLM",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    return model, tokenizer


def generate(model: MlxOpenELMForCausalLM, tokenizer: AutoTokenizer, args):
    print(args.prompt)
    inputs = tokenizer(args.prompt, return_tensors="np", truncation=True)

    inputs = {key: mx.array(v) for key, v in inputs.items()}
    print(inputs["input_ids"][0])

    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(inputs, max_length=args.max_tokens, temp=args.temp):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)
            prompt_processing = toc("Prompt processing", start)

        if len(tokens) >= args.max_tokens:
            break

        elif (len(tokens) % args.write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
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
    parser = argparse.ArgumentParser(description="Llama inference script")
    parser.add_argument(
        "--model-name",
        help="The model name to load",
        default="apple/OpenELM-1_1B-Instruct",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model.",
        default="Who is your daddy and what does he do?",
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=100, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=5, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="The sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()

    mx.random.seed(args.seed)
    mx.set_default_device(mx.gpu)

    model, tokenizer = load_model(
        args.model_name,
        MlxOpenELMForCausalLM,
    )

    generate(model, tokenizer, args)
