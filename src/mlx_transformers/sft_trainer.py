import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

import mlx.core as mx
import numpy as np

from .lora import LoraConfig, apply_lora, has_lora_layers, save_lora_adapters
from .trainer import Trainer, TrainingArguments


def _pad_sequences(
    sequences: list[np.ndarray],
    padding_value: int,
    pad_to_multiple_of: Optional[int] = None,
) -> np.ndarray:
    if not sequences:
        raise ValueError("Cannot pad an empty batch.")

    max_length = max(len(sequence) for sequence in sequences)
    if pad_to_multiple_of is not None and max_length % pad_to_multiple_of != 0:
        max_length = (
            (max_length + pad_to_multiple_of - 1) // pad_to_multiple_of
        ) * pad_to_multiple_of

    batch = np.full((len(sequences), max_length), padding_value, dtype=np.int32)
    for index, sequence in enumerate(sequences):
        batch[index, : len(sequence)] = sequence
    return batch


@dataclass
class SFTConfig(TrainingArguments):
    max_length: int = 1024
    dataset_text_field: str = "text"
    completion_only_loss: bool = False
    packing: bool = False
    pad_to_multiple_of: Optional[int] = None


@dataclass
class DataCollatorForLanguageModeling:
    pad_token_id: int
    completion_only_loss: bool = False
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, examples: list[Mapping[str, Any]]) -> dict[str, mx.array]:
        input_ids = [
            np.asarray(example["input_ids"], dtype=np.int32) for example in examples
        ]
        labels = [
            np.asarray(example.get("labels", example["input_ids"]), dtype=np.int32)
            for example in examples
        ]
        attention_mask = [np.ones(len(ids), dtype=np.int32) for ids in input_ids]

        batch = {
            "input_ids": mx.array(
                _pad_sequences(input_ids, self.pad_token_id, self.pad_to_multiple_of)
            ),
            "attention_mask": mx.array(
                _pad_sequences(attention_mask, 0, self.pad_to_multiple_of)
            ),
            "labels": mx.array(_pad_sequences(labels, -100, self.pad_to_multiple_of)),
        }

        if self.completion_only_loss and "completion_mask" in examples[0]:
            completion_masks = [
                np.asarray(example["completion_mask"], dtype=np.int32)
                for example in examples
            ]
            completion_masks = _pad_sequences(
                completion_masks,
                0,
                self.pad_to_multiple_of,
            )
            batch["labels"] = mx.where(
                mx.array(completion_masks) == 1,
                batch["labels"],
                mx.full(batch["labels"].shape, -100, dtype=mx.int32),
            )

        return batch


class SFTTrainer(Trainer):
    def __init__(
        self,
        model,
        args: Optional[SFTConfig] = None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        tokenizer=None,
        data_collator=None,
        formatting_func: Optional[Callable[[Mapping[str, Any]], str]] = None,
        lora_config: Optional[LoraConfig] = None,
        compute_metrics=None,
        callbacks=None,
    ) -> None:
        self.args = args or SFTConfig()
        self.processing_class = processing_class or tokenizer
        self.formatting_func = formatting_func
        self.lora_config = lora_config

        if self.args.packing:
            raise NotImplementedError("Sequence packing is not implemented yet.")

        if self.lora_config is not None:
            model = apply_lora(model, self.lora_config)

        train_dataset = self._prepare_dataset(train_dataset)
        eval_dataset = self._prepare_dataset(eval_dataset)

        if data_collator is None:
            pad_token_id = self._resolve_pad_token_id()
            data_collator = DataCollatorForLanguageModeling(
                pad_token_id=pad_token_id,
                completion_only_loss=self.args.completion_only_loss,
                pad_to_multiple_of=self.args.pad_to_multiple_of,
            )

        super().__init__(
            model=model,
            args=self.args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    def save_model(self, output_dir: Optional[str] = None) -> Path:
        checkpoint_dir = Path(output_dir or self.args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if self.lora_config is not None and has_lora_layers(self.model):
            save_lora_adapters(self.model, checkpoint_dir, self.lora_config)
            if self.optimizer is not None:
                mx.save_safetensors(
                    str(checkpoint_dir / "optimizer.safetensors"),
                    dict(self._flatten_optimizer_state()),
                )
            with (checkpoint_dir / "training_args.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(self._json_ready(self.args.__dict__), f, indent=2)
            with (checkpoint_dir / "trainer_state.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(self._json_ready(self.state.__dict__), f, indent=2)
            return checkpoint_dir

        return super().save_model(str(checkpoint_dir))

    def _prepare_dataset(self, dataset):
        if dataset is None:
            return None

        sample = self._peek_example(dataset)
        if sample is None or "input_ids" in sample:
            return dataset

        if self.processing_class is None:
            raise ValueError(
                "SFTTrainer requires a tokenizer or processing_class for raw-text datasets."
            )

        if "prompt" in sample and "completion" in sample:
            return self._map_dataset(dataset, self._prepare_prompt_completion_example)
        if "messages" in sample:
            return self._map_dataset(dataset, self._prepare_messages_example)
        if self.formatting_func is not None or self.args.dataset_text_field in sample:
            return self._map_dataset(dataset, self._prepare_text_example)

        raise ValueError(
            "Unsupported SFT dataset format. Expected tokenized `input_ids`, "
            f"`{self.args.dataset_text_field}`, `prompt`/`completion`, or `messages`."
        )

    def _prepare_text_example(self, example: Mapping[str, Any]) -> dict[str, Any]:
        text = (
            self.formatting_func(example)
            if self.formatting_func is not None
            else example[self.args.dataset_text_field]
        )
        return {"input_ids": self._tokenize_text(self._normalize_text(text))}

    def _prepare_messages_example(self, example: Mapping[str, Any]) -> dict[str, Any]:
        if not hasattr(self.processing_class, "apply_chat_template"):
            raise ValueError(
                "The provided tokenizer does not support apply_chat_template."
            )

        try:
            text = self.processing_class.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
        except TypeError:
            text = self.processing_class.apply_chat_template(
                example["messages"],
                tokenize=False,
            )

        return {"input_ids": self._tokenize_text(text)}

    def _prepare_prompt_completion_example(
        self, example: Mapping[str, Any]
    ) -> dict[str, Any]:
        prompt = self._normalize_text(example["prompt"])
        completion = self._normalize_text(example["completion"])

        prompt_ids = self._tokenize_text(prompt, add_special_tokens=True)
        completion_ids = self._tokenize_text(completion, add_special_tokens=False)
        input_ids = prompt_ids + completion_ids

        if self.args.max_length is not None and len(input_ids) > self.args.max_length:
            input_ids = input_ids[: self.args.max_length]

        result = {"input_ids": input_ids}
        if self.args.completion_only_loss:
            completion_mask = ([0] * len(prompt_ids)) + ([1] * len(completion_ids))
            result["completion_mask"] = completion_mask[: len(input_ids)]

        return result

    def _map_dataset(
        self, dataset, transform: Callable[[Mapping[str, Any]], dict[str, Any]]
    ):
        if hasattr(dataset, "map"):
            remove_columns = getattr(dataset, "column_names", None)
            return dataset.map(
                transform,
                remove_columns=list(remove_columns)
                if remove_columns is not None
                else None,
            )

        if self._is_indexable_dataset(dataset):
            return [transform(dataset[index]) for index in range(len(dataset))]

        return [transform(example) for example in dataset]

    def _peek_example(self, dataset):
        if self._is_indexable_dataset(dataset) and len(dataset) > 0:
            return dataset[0]

        iterator = iter(dataset)
        try:
            return next(iterator)
        except StopIteration:
            return None

    def _tokenize_text(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoded = self.processing_class(
            text,
            add_special_tokens=add_special_tokens,
            truncation=self.args.max_length is not None,
            max_length=self.args.max_length,
        )
        input_ids = encoded["input_ids"]
        if isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()
        return list(input_ids)

    def _resolve_pad_token_id(self) -> int:
        if self.processing_class is None:
            raise ValueError(
                "A tokenizer or explicit data_collator is required for SFTTrainer."
            )

        pad_token_id = getattr(self.processing_class, "pad_token_id", None)
        if pad_token_id is not None:
            return int(pad_token_id)

        eos_token_id = getattr(self.processing_class, "eos_token_id", None)
        if eos_token_id is not None:
            if hasattr(self.processing_class, "pad_token") and hasattr(
                self.processing_class, "eos_token"
            ):
                self.processing_class.pad_token = self.processing_class.eos_token
            self.processing_class.pad_token_id = eos_token_id
            return int(eos_token_id)

        raise ValueError("Tokenizer must define pad_token_id or eos_token_id.")

    def _flatten_optimizer_state(self):
        from mlx.utils import tree_flatten

        return tree_flatten(self.optimizer.state)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)):
            return "\n".join(str(item) for item in value)
        return str(value)
