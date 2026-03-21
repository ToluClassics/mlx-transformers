from typing import Set

from transformers import GenerationConfig


def get_eos_token_ids(model_name: str, tokenizer) -> Set[int]:
    eos_token_ids = None

    try:
        generation_config = GenerationConfig.from_pretrained(model_name)
        eos_token_ids = generation_config.eos_token_id
    except (OSError, ValueError):
        eos_token_ids = None

    if eos_token_ids is None:
        eos_token_ids = tokenizer.eos_token_id

    if eos_token_ids is None:
        return set()
    if isinstance(eos_token_ids, int):
        return {eos_token_ids}
    return set(eos_token_ids)
