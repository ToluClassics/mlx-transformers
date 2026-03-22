import argparse
import time
from typing import Tuple

import mlx.core as mx
from transformers import AutoConfig, AutoTokenizer

from common import get_eos_token_ids
from mlx_transformers.models import Qwen3ForCausalLM as MlxQwen3ForCausalLM


def tic():
    "Return generation time in seconds"
    return time.time()


def toc(msg, start):
    "Return generation time in seconds and a message"
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def load_model(model_name: str) -> Tuple[MlxQwen3ForCausalLM, AutoTokenizer]:
    config = AutoConfig.from_pretrained(model_name)

    model = MlxQwen3ForCausalLM(config)
    model.from_pretrained(model_name)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def generate(model: MlxQwen3ForCausalLM, tokenizer: AutoTokenizer, args):
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
    parser = argparse.ArgumentParser(description="Qwen3 inference script")
    parser.add_argument(
        "--model-name",
        help="The model name to load",
        default="Qwen/Qwen3-0.6B",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model.",
        default="Explain grouped-query attention in one paragraph.",
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

    args = parser.parse_args()

    mx.random.seed(args.seed)
    mx.set_default_device(mx.gpu)

    model, tokenizer = load_model(args.model_name)

    generate(model, tokenizer, args)
