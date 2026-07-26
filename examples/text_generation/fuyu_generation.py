import argparse
import time
from typing import Tuple

from PIL import Image
import mlx.core as mx
from transformers import FuyuProcessor, FuyuConfig

from mlx_transformers.models import FuyuForCausalLM as MlxFuyuForCausalLM


def tic():
    "Return generation time in seconds"
    return time.time()


def toc(msg, start):
    "Return generation time in seconds and a message"
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def load_model(
    model_name: str, mlx_model_class
) -> Tuple[MlxFuyuForCausalLM, FuyuProcessor]:
    """
    Load a llama model and tokenizer from the given model name and weights.

    Args:
        model_name (str): Name of the llama model to load
        model_weights (str): Path to the model weights
        hgf_model_class: Huggingface model class
        mlx_model_class: Mlx model class

    """
    config = FuyuConfig.from_pretrained(model_name)

    model = mlx_model_class(config)
    model.from_pretrained(model_name)

    processor = FuyuProcessor.from_pretrained(model_name)

    return model, processor


def generate(
    model: MlxFuyuForCausalLM,
    processor: FuyuProcessor,
    prompt: str,
    image: str,
    max_tokens: int = 10,
    temp: float = 1.0,
    write_every: int = 1,
):
    inputs = processor(text=prompt, images=image, return_tensors="np")
    converted_inputs = {}

    for key, value in inputs.items():
        if isinstance(value, list):
            val_list = []
            for v in value:
                val_list.append(mx.array(v))
            converted_inputs[key] = val_list
        else:
            converted_inputs[key] = mx.array(value)

    skip = 0
    prompt_processing = None
    tokens = []
    start = tic()
    for token in model.generate(
        converted_inputs,
        max_new_tokens=max_tokens,
        temp=temp,
    ):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            prompt_processing = toc("Prompt processing", start)

        if (len(tokens) % write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            s = processor.decode([t.item() for t in tokens])
            print(s[skip:], end="", flush=True)
            skip = len(s)

    full_gen = toc("Full generation", start)
    s = processor.decode([t.item() for t in tokens])
    print(s, flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate text from an image with Fuyu."
    )
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--image_path", required=True)
    parser.add_argument(
        "--prompt",
        default="Generate a concise caption for this image.\n",
    )
    parser.add_argument("--max_tokens", type=int, default=32)
    parser.add_argument("--temp", type=float, default=0.0)
    args = parser.parse_args()

    image = Image.open(args.image_path).convert("RGB")
    mx.set_default_device(mx.gpu)

    model, processor = load_model(args.model_name, MlxFuyuForCausalLM)

    generate(
        model,
        processor,
        args.prompt,
        image,
        max_tokens=args.max_tokens,
        temp=args.temp,
    )
