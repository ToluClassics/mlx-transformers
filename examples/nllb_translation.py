"Utils for translation models"

import argparse
import os
import time
from typing import List, Tuple

import mlx.core as mx
import numpy as np
import transformers
from flores200_codes import FLORES_CODES
from transformers import M2M100Config, NllbTokenizer, PreTrainedTokenizerBase

from mlx_transformers.models import (
    M2M100ForConditionalGeneration as MlxM2M100ForConditionalGeneration,
)


def top_p_sampling(logits: mx.array, top_p: float, temperature: float) -> mx.array:
    """
    Apply top-p (nucleus) sampling to logits.

    Args:
        logits: The logits from the model's output.
        top_p: The cumulative probability threshold for top-p filtering.
        temperature: Temperature parameter for softmax distribution reshaping.
    Returns:
        token selected based on the top-p criterion.
    """
    # referenced implementation from https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py#L449-L460
    probs = mx.softmax(logits / temperature, axis=-1)

    # sort probs in ascending order
    sorted_indices = mx.argsort(probs, axis=-1)
    sorted_probs = probs[..., sorted_indices.squeeze(0)]

    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

    # select tokens with cumulative probs below threshold
    top_probs = mx.where(
        cumulative_probs > 1 - top_p,
        sorted_probs,
        mx.zeros_like(sorted_probs),
    )

    sorted_token = mx.random.categorical(mx.log(top_probs))
    token = sorted_indices.squeeze(0)[sorted_token]

    return token


def load_mlx_nllb_model(
    model_name: str, source_language: str, target_language: str
) -> Tuple[MlxM2M100ForConditionalGeneration, PreTrainedTokenizerBase, int]:
    """
    Load a nllb model and tokenizer from the given model name and weights path.
    """
    if source_language in FLORES_CODES and target_language in FLORES_CODES:
        source_language = FLORES_CODES[source_language]
        target_language = FLORES_CODES[target_language]

    config = M2M100Config.from_pretrained(model_name)

    model = MlxM2M100ForConditionalGeneration(config)
    model.from_pretrained(model_name)

    tokenizer = NllbTokenizer.from_pretrained(
        model_name, src_lang=source_language, tgt_lang=target_language
    )

    tgt_token_id = tokenizer.convert_tokens_to_ids("yor_Latn")
    return model, tokenizer, tgt_token_id


def sample(logits: mx.array, temp: float, top_p: float) -> Tuple[mx.array, float]:
    """
    Sample a token from the logits.

    Args:
        logits (mx.array): Logits from the model
        temp (float): Temperature for sampling
        top_p (float): Top-p sampling value

    Returns:
        Tuple[mx.array, float]: Tuple of the token and its probability
    """
    softmax_logits = mx.softmax(logits)

    if temp == 0:
        token = mx.argmax(logits, axis=-1)
    else:
        if top_p > 0 and top_p < 1.0:
            token = top_p_sampling(logits, top_p, temp)
        else:
            token = mx.random.categorical(logits * (1 / temp))

    prob = softmax_logits[0, token]
    return token, prob


def run_translation_mlx(
    model: MlxM2M100ForConditionalGeneration,
    tokenizer: PreTrainedTokenizerBase,
    input_sentences: List[str],
    target_language_token: int,
    max_generation_tokens: int,
    temp: float = 0.0,
    top_p: float = 1.0,
    verbose: bool = False,
) -> List[str]:
    """
    Run Translation using the MLX model.

    Returns:
         List[str]: List of translated sentences
    """

    bsz = len(input_sentences)
    tokens = tokenizer(input_sentences, return_tensors="np", padding="longest")

    tokens = {key: mx.array(v) for key, v in tokens.items()}

    decoder_input_ids = mx.array(
        [[model.config.eos_token_id, target_language_token]] * bsz
    )
    decoder_input_mask = mx.array([[1, 1]] * bsz)

    encoder_tokens = model.encode(tokens["input_ids"], tokens["attention_mask"])

    start_time = time.time()  # Start measuring time

    for _ in range(max_generation_tokens):
        outputs = model.decode(
            decoder_input_ids,
            decoder_input_mask,
            encoder_tokens,
            tokens["attention_mask"],
            None,
        )
        logits = model.lm_head(outputs)[:, -1, :]

        next_token, _ = sample(logits, temp, top_p)

        decoder_input_ids = mx.concatenate(
            [decoder_input_ids, next_token.reshape(-1, 1)], axis=1
        )
        decoder_input_mask = mx.concatenate(
            [decoder_input_mask, mx.ones((bsz, 1))], axis=1
        )

        if next_token[0] == model.config.eos_token_id:
            break

    end_time = time.time()  # Stop measuring time

    if verbose:
        total_tokens = decoder_input_ids.size - (bsz * 2)
        total_time = end_time - start_time
        token_per_sec = total_tokens / total_time
        print(f"Generated {total_tokens} tokens in {total_time} seconds.")
        print(f"Token/s: {token_per_sec}")

    translated_sentences = tokenizer.batch_decode(
        np.array(decoder_input_ids), skip_special_tokens=True
    )

    return translated_sentences


def main(args):

    model, tokenizer, tgt_token_id = load_mlx_nllb_model(
        model_name=args.model_name,
        source_language=args.source_language,
        target_language=args.target_language,
    )

    text_to_translate = args.text_to_translate

    output = run_translation_mlx(
        model=model,
        tokenizer=tokenizer,
        input_sentences=[text_to_translate],
        target_language_token=tgt_token_id,
        max_generation_tokens=args.max_generation_tokens,
        verbose=True,
    )

    print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--source_language", type=str, required=True)
    parser.add_argument("--target_language", type=str, required=True)
    parser.add_argument(
        "--text_to_translate", type=str, default="Let us translate text to Yoruba"
    )
    parser.add_argument("--max_generation_tokens", type=int, default=20)
    args = parser.parse_args()
    main(args)
