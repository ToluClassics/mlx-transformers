import argparse
import os
import time
from pathlib import Path

import mlx.core as mx
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

from mlx_transformers.models import LlamaForCausalLM as MlxLlamaForCausalLM
from mlx_transformers.models.utils import convert


def tic():
    return time.time()


def toc(msg, start):
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def load_model(model_name: str, model_weights: str, hgf_model_class, mlx_model_class):
    config = LlamaConfig.from_pretrained(model_name)
    current_directory = os.path.dirname(os.path.realpath(__file__))

    model = mlx_model_class(config)
    model.load_weights(model_weights, strict=False)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def generate(model, tokenizer, args):
    print(args.prompt)
    inputs = tokenizer(args.prompt, return_tensors="np", truncation=True)

    inputs = {key: mx.array(v) for key, v in inputs.items()}
    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(inputs, args.temp):
        tokens.append(token)

        print(tokenizer.decode(inputs["input_ids"][0].tolist() + [token.item()]))

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
        default="meta-llama/Llama-2-7b-hf",
    )
    parser.add_argument(
        "--model-path",
        help="Path to the model weights",
        default="mlx_model",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model. Ignored when --few-shot is provided.",
        default="In the beginning the Universe was created.",
    )
    parser.add_argument(
        "--few-shot",
        help="Read a few shot prompt from a file (as in `sample_prompt.txt`).",
        required=False,
    )
    parser.add_argument(
        "--max-tokens", "-m", type=int, default=100, help="How many tokens to generate"
    )
    parser.add_argument(
        "--write-every", type=int, default=1, help="After how many tokens to detokenize"
    )
    parser.add_argument(
        "--temp", type=float, default=0.0, help="The sampling temperature"
    )
    parser.add_argument("--seed", type=int, default=0, help="The PRNG seed")

    args = parser.parse_args()

    mx.random.seed(args.seed)

    model, tokenizer = load_model(
        args.model_name, args.model_path, LlamaForCausalLM, MlxLlamaForCausalLM
    )

    generate(model, tokenizer, args)
