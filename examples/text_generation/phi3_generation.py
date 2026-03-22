import argparse
import os
import time
from typing import Tuple

import mlx.core as mx
from transformers import AutoTokenizer, AutoConfig

from mlx_transformers.models import Phi3ForCausalLM as MlxPhi3ForCausalLM
from common import get_eos_token_ids

DEFAULT_SYSTEM_PROMPT = (
    "You are Phi, a language model trained by Microsoft to help users. "
    "Your role as an assistant involves thoroughly exploring questions through "
    "a systematic thinking process before providing the final precise and "
    "accurate solutions. This requires engaging in a comprehensive cycle of "
    "analysis, summarizing, exploration, reassessment, reflection, "
    "backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and "
    "Solution using the specified format: <think> {Thought section} </think> "
    "{Solution section}. In the Thought section, detail your reasoning "
    "process in steps. Each step should include detailed considerations such "
    "as analysing questions, summarizing relevant findings, brainstorming new "
    "ideas, verifying the accuracy of the current steps, refining any "
    "errors, and revisiting previous steps. In the Solution section, based "
    "on various attempts, explorations, and reflections from the Thought "
    "section, systematically present the final solution that you deem "
    "correct. The Solution section should be logical, accurate, and concise "
    "and detail necessary steps needed to reach the conclusion. Now, try to "
    "solve the following question through the above guidelines:"
)


def tic():
    "Return generation time in seconds"
    return time.time()


def toc(msg, start):
    "Return generation time in seconds and a message"
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def load_model(
    model_name: str, mlx_model_class, fp16: bool = False
) -> Tuple[MlxPhi3ForCausalLM, AutoTokenizer]:
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
        trust_remote_code=True,
        float16=fp16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


def generate(model: MlxPhi3ForCausalLM, tokenizer: AutoTokenizer, args):
    print(args.prompt)
    messages = [
        {"role": "system", "content": args.system_prompt},
        {"role": "user", "content": args.prompt},
    ]
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
    parser = argparse.ArgumentParser(description="Phi3 inference script")
    parser.add_argument(
        "--model-name",
        help="The model name to load",
        default="microsoft/Phi-3-mini-4k-instruct",
    )
    parser.add_argument(
        "--prompt",
        help="The message to be processed by the model.",
        default="In the beginning the Universe was created.",
    )
    parser.add_argument(
        "--system-prompt",
        help="The system prompt prepended to Phi requests.",
        default=DEFAULT_SYSTEM_PROMPT,
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
    parser.add_argument(
        "--fp16", action="store_true", help="Use mixed precision for inference"
    )

    args = parser.parse_args()

    mx.random.seed(args.seed)
    mx.set_default_device(mx.gpu)

    model, tokenizer = load_model(args.model_name, MlxPhi3ForCausalLM)

    generate(model, tokenizer, args)
