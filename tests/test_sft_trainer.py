import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datasets import Dataset

from src.mlx_transformers.lora import LoraConfig, LoRALinear
from src.mlx_transformers.models.modelling_outputs import CausalLMOutputWithPast
from src.mlx_transformers.sft_trainer import (
    DataCollatorForLanguageModeling,
    SFTConfig,
    SFTTrainer,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __init__(self):
        self._vocab = {
            self.pad_token: self.pad_token_id,
            self.eos_token: self.eos_token_id,
        }

    def __call__(
        self,
        text,
        add_special_tokens=True,
        truncation=False,
        max_length=None,
    ):
        if isinstance(text, list):
            return {
                "input_ids": [
                    self._encode(item, add_special_tokens, truncation, max_length)
                    for item in text
                ]
            }
        return {
            "input_ids": self._encode(text, add_special_tokens, truncation, max_length)
        }

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=False
    ):
        rendered = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        if add_generation_prompt:
            rendered += "\nassistant:"
        if tokenize:
            return self(rendered)["input_ids"]
        return rendered

    def _encode(self, text, add_special_tokens, truncation, max_length):
        tokens = str(text).split()
        input_ids = []
        for token in tokens:
            if token not in self._vocab:
                self._vocab[token] = len(self._vocab)
            input_ids.append(self._vocab[token])
        if add_special_tokens:
            input_ids.append(self.eos_token_id)
        if truncation and max_length is not None:
            input_ids = input_ids[:max_length]
        return input_ids


class TinyCausalLM(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=16):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def __call__(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.embed_tokens(input_ids.astype(mx.int32))
        logits = self.lm_head(hidden_states).astype(mx.float32)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].astype(mx.int32)
            valid_mask = (shift_labels != -100).astype(shift_logits.dtype)
            safe_labels = mx.where(shift_labels != -100, shift_labels, 0)
            token_loss = nn.losses.cross_entropy(
                shift_logits,
                safe_labels,
                reduction="none",
            )
            loss = mx.sum(token_loss * valid_mask) / mx.maximum(mx.sum(valid_mask), 1.0)

        return CausalLMOutputWithPast(loss=loss, logits=logits)


class TinyLoraAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


class TinyLoraLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.self_attn = TinyLoraAttention(hidden_size)
        self.mlp = nn.Linear(hidden_size, hidden_size, bias=False)

    def __call__(self, hidden_states):
        hidden_states = self.self_attn.o_proj(
            self.self_attn.q_proj(hidden_states) + self.self_attn.v_proj(hidden_states)
        )
        return self.mlp(hidden_states)


class TinyLoraBackbone(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=16, num_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = [TinyLoraLayer(hidden_size) for _ in range(num_layers)]

    def __call__(self, input_ids):
        hidden_states = self.embed_tokens(input_ids.astype(mx.int32))
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TinyLoraCausalLM(nn.Module):
    def __init__(self, vocab_size=128, hidden_size=16, num_layers=2):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = TinyLoraBackbone(vocab_size, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def __call__(self, input_ids, attention_mask=None, labels=None):
        hidden_states = self.model(input_ids)
        logits = self.lm_head(hidden_states).astype(mx.float32)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].astype(mx.int32)
            valid_mask = (shift_labels != -100).astype(shift_logits.dtype)
            safe_labels = mx.where(shift_labels != -100, shift_labels, 0)
            token_loss = nn.losses.cross_entropy(
                shift_logits,
                safe_labels,
                reduction="none",
            )
            loss = mx.sum(token_loss * valid_mask) / mx.maximum(mx.sum(valid_mask), 1.0)

        return CausalLMOutputWithPast(loss=loss, logits=logits)


class TestSFTTrainer(unittest.TestCase):
    def test_language_modeling_collator_pads_and_masks(self):
        collator = DataCollatorForLanguageModeling(pad_token_id=0)
        batch = collator([{"input_ids": [1, 2, 3]}, {"input_ids": [4, 5]}])

        self.assertTrue(
            np.array_equal(
                np.array(batch["input_ids"]), np.array([[1, 2, 3], [4, 5, 0]])
            )
        )
        self.assertTrue(
            np.array_equal(
                np.array(batch["labels"]),
                np.array([[1, 2, 3], [4, 5, -100]]),
            )
        )

    def test_language_modeling_collator_respects_completion_mask(self):
        collator = DataCollatorForLanguageModeling(
            pad_token_id=0,
            completion_only_loss=True,
        )
        batch = collator(
            [
                {"input_ids": [1, 2, 3], "completion_mask": [0, 1, 1]},
                {"input_ids": [4, 5], "completion_mask": [0, 1]},
            ]
        )

        self.assertTrue(
            np.array_equal(
                np.array(batch["labels"]),
                np.array([[-100, 2, 3], [-100, 5, -100]]),
            )
        )

    def test_sft_trainer_prepares_text_dataset_and_trains(self):
        tokenizer = FakeTokenizer()
        model = TinyCausalLM()
        dataset = Dataset.from_dict(
            {
                "text": [
                    "great movie with heart",
                    "bad movie and boring ending",
                    "fun performances and sharp dialogue",
                    "predictable plot but solid acting",
                ]
            }
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                args=SFTConfig(
                    output_dir=tmpdir,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    num_train_epochs=1,
                    max_steps=1,
                    logging_steps=1,
                    max_length=16,
                    seed=0,
                ),
                processing_class=tokenizer,
                train_dataset=dataset,
                eval_dataset=dataset,
            )

            self.assertIn("input_ids", trainer.train_dataset.column_names)
            self.assertNotIn("text", trainer.train_dataset.column_names)

            train_result = trainer.train()
            eval_metrics = trainer.evaluate()

            self.assertEqual(train_result["global_step"], 1.0)
            self.assertIn("train_loss", train_result)
            self.assertIn("eval_loss", eval_metrics)

    def test_sft_trainer_prompt_completion_builds_completion_mask(self):
        tokenizer = FakeTokenizer()
        model = TinyCausalLM()
        dataset = [
            {"prompt": "Question:", "completion": "Answer one"},
            {"prompt": "Prompt:", "completion": "Reply two"},
        ]

        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                output_dir="unused",
                per_device_train_batch_size=2,
                max_steps=1,
                completion_only_loss=True,
                max_length=16,
            ),
            processing_class=tokenizer,
            train_dataset=dataset,
        )

        first_example = trainer.train_dataset[0]
        self.assertIn("completion_mask", first_example)
        self.assertIn(0, first_example["completion_mask"])
        self.assertIn(1, first_example["completion_mask"])

        batch = trainer.data_collator(trainer.train_dataset[:2])
        labels = np.array(batch["labels"])
        self.assertTrue(np.any(labels == -100))

    def test_sft_trainer_with_lora_saves_adapters(self):
        tokenizer = FakeTokenizer()
        model = TinyLoraCausalLM()
        dataset = Dataset.from_dict(
            {"text": ["alpha beta gamma", "delta epsilon zeta", "eta theta iota"]}
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = SFTTrainer(
                model=model,
                args=SFTConfig(
                    output_dir=tmpdir,
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    num_train_epochs=1,
                    max_steps=1,
                    max_length=16,
                    seed=0,
                ),
                processing_class=tokenizer,
                train_dataset=dataset,
                eval_dataset=dataset,
                lora_config=LoraConfig(num_layers=1, r=4),
            )

            self.assertTrue(
                isinstance(trainer.model.model.layers[-1].self_attn.q_proj, LoRALinear)
            )

            trainer.train()

            self.assertTrue((Path(tmpdir) / "adapters.safetensors").exists())
            self.assertTrue((Path(tmpdir) / "adapter_config.json").exists())


if __name__ == "__main__":
    unittest.main()
