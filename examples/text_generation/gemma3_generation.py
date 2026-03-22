import argparse
import io
import time
import urllib.request
from typing import Tuple

import mlx.core as mx
from PIL import Image
import torch
from transformers import AutoConfig, AutoProcessor

from common import get_eos_token_ids
from mlx_transformers.models import (
    Gemma3ForConditionalGeneration as MlxGemma3ForConditionalGeneration,
)


def tic():
    "Return generation time in seconds"
    return time.time()


def toc(msg, start):
    "Return generation time in seconds and a message"
    end = time.time()
    return f"[INFO] {msg}: {end - start:.3f} s"


def load_model(
    model_name: str,
    args,
) -> Tuple[MlxGemma3ForConditionalGeneration, AutoProcessor]:
    config = AutoConfig.from_pretrained(model_name)

    model = MlxGemma3ForConditionalGeneration(config)
    model.from_pretrained(
        model_name,
        quantize=args.quantize,
        group_size=args.group_size,
        bits=args.bits,
        mode=args.mode,
        quantize_input=args.quantize_input,
    )

    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def load_image(args) -> Image.Image:
    if args.image_path:
        image = Image.open(args.image_path)
    else:
        with urllib.request.urlopen(args.image_url) as response:
            image = Image.open(io.BytesIO(response.read()))

    return image.convert("RGB")


def convert_inputs_to_mx(inputs):
    converted_inputs = {}
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            converted_inputs[key] = mx.array(value.detach().cpu().numpy())
        else:
            converted_inputs[key] = mx.array(value)
    return converted_inputs


def generate(
    model: MlxGemma3ForConditionalGeneration,
    processor: AutoProcessor,
    args,
):
    image = load_image(args)
    print(args.prompt)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = convert_inputs_to_mx(inputs)
    eos_token_ids = get_eos_token_ids(args.model_name, processor.tokenizer)

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
            decoded = processor.tokenizer.decode([t.item() for t in tokens])
            print(decoded[skip:], end="", flush=True)
            skip = len(decoded)

    mx.eval(tokens)
    full_gen = toc("Full generation", start)
    decoded = processor.tokenizer.decode([t.item() for t in tokens])
    print(decoded[skip:], flush=True)
    print("------")
    print(prompt_processing)
    print(full_gen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma3 inference script")
    parser.add_argument(
        "--model-name",
        help="The model name to load",
        default="google/gemma-3-4b-it",
    )
    parser.add_argument(
        "--prompt",
        help="The text prompt paired with the image.",
        default="What is happening in this image?",
    )
    parser.add_argument(
        "--image-path",
        help="Path to a local image. If omitted, --image-url is used.",
        default=None,
    )
    parser.add_argument(
        "--image-url",
        help="Image URL to fetch when --image-path is not provided.",
        default="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg",
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

    if not args.image_path and not args.image_url:
        raise ValueError("Pass either --image-path or --image-url.")

    mx.random.seed(args.seed)
    mx.set_default_device(mx.gpu)

    model, processor = load_model(args.model_name, args)
    generate(model, processor, args)
