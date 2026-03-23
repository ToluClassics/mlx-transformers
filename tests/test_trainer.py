import tempfile
import unittest
from pathlib import Path

import numpy as np
import mlx.core as mx
import mlx.nn as nn
from transformers import BertConfig

from src.mlx_transformers.models import BertForSequenceClassification
from src.mlx_transformers.models.modelling_outputs import SequenceClassifierOutput
from src.mlx_transformers.trainer import (
    Trainer,
    TrainingArguments,
    default_data_collator,
)


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 2)

    def __call__(self, input_ids, attention_mask=None):
        logits = self.classifier(input_ids.astype(mx.float32))
        return SequenceClassifierOutput(logits=logits)


class IndexableDataset:
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class TestTrainer(unittest.TestCase):
    def _tiny_config(self) -> BertConfig:
        return BertConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=16,
            num_labels=2,
        )

    def _dataset(self):
        return [
            {
                "input_ids": np.array([1, 2, 3, 4], dtype=np.int32),
                "attention_mask": np.array([1, 1, 1, 1], dtype=np.int32),
                "token_type_ids": np.array([0, 0, 0, 0], dtype=np.int32),
                "labels": 0,
            },
            {
                "input_ids": np.array([4, 3, 2, 1], dtype=np.int32),
                "attention_mask": np.array([1, 1, 1, 1], dtype=np.int32),
                "token_type_ids": np.array([0, 0, 0, 0], dtype=np.int32),
                "labels": 1,
            },
            {
                "input_ids": np.array([1, 1, 2, 2], dtype=np.int32),
                "attention_mask": np.array([1, 1, 1, 1], dtype=np.int32),
                "token_type_ids": np.array([0, 0, 0, 0], dtype=np.int32),
                "labels": 0,
            },
            {
                "input_ids": np.array([2, 2, 1, 1], dtype=np.int32),
                "attention_mask": np.array([1, 1, 1, 1], dtype=np.int32),
                "token_type_ids": np.array([0, 0, 0, 0], dtype=np.int32),
                "labels": 1,
            },
        ]

    def test_default_data_collator_stacks_arrays(self):
        batch = default_data_collator(self._dataset()[:2])

        self.assertEqual(batch["input_ids"].shape, (2, 4))
        self.assertEqual(batch["attention_mask"].shape, (2, 4))
        self.assertEqual(type(batch["labels"]), mx.array)

    def test_trainer_runs_train_eval_and_checkpoint_cycle(self):
        model = BertForSequenceClassification(self._tiny_config())
        dataset = self._dataset()

        def compute_metrics(prediction):
            predicted = prediction.predictions.argmax(axis=-1)
            labels = prediction.label_ids
            return {"accuracy": float((predicted == labels).mean())}

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=tmpdir,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    num_train_epochs=1,
                    max_steps=2,
                    logging_steps=1,
                    eval_steps=1,
                    save_steps=1,
                    seed=0,
                ),
                train_dataset=dataset,
                eval_dataset=dataset,
                compute_metrics=compute_metrics,
            )

            train_result = trainer.train()
            metrics = trainer.evaluate()

            self.assertEqual(train_result["global_step"], 2.0)
            self.assertIn("train_loss", train_result)
            self.assertIn("eval_loss", metrics)
            self.assertIn("accuracy", metrics)

            output_dir = Path(tmpdir)
            self.assertTrue((output_dir / "model.safetensors").exists())
            self.assertTrue((output_dir / "training_args.json").exists())
            self.assertTrue((output_dir / "trainer_state.json").exists())
            self.assertTrue(
                (output_dir / "checkpoint-1" / "model.safetensors").exists()
            )

    def test_custom_loss_path_strips_labels_and_is_used_for_eval(self):
        model = TinyClassifier()
        dataset = self._dataset()[:2]

        def compute_loss_func(model, inputs, outputs):
            return nn.losses.cross_entropy(outputs.logits, inputs["labels"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=tmpdir,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    num_train_epochs=1,
                    max_steps=1,
                    logging_steps=1,
                    seed=0,
                ),
                train_dataset=dataset,
                eval_dataset=dataset,
                compute_loss_func=compute_loss_func,
            )

            trainer.train()
            batch = default_data_collator(dataset)
            expected_loss = trainer.compute_loss(model, batch)
            mx.eval(expected_loss)

            metrics = trainer.evaluate()

            self.assertIn("eval_loss", metrics)
            self.assertAlmostEqual(
                metrics["eval_loss"],
                float(expected_loss.item()),
                places=5,
            )

    def test_extract_predictions_ignores_single_loss_tuple(self):
        self.assertIsNone(Trainer._extract_predictions((mx.array(1.0),)))

    def test_indexable_non_sequence_dataset_is_shuffled_like_map_style_data(self):
        model = TinyClassifier()
        dataset = IndexableDataset(self._dataset()[:2])

        def compute_loss_func(model, inputs, outputs):
            return nn.losses.cross_entropy(outputs.logits, inputs["labels"])

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=tmpdir,
                    per_device_train_batch_size=2,
                    num_train_epochs=1,
                    max_steps=1,
                    seed=0,
                ),
                train_dataset=dataset,
                compute_loss_func=compute_loss_func,
            )

            train_result = trainer.train()

            self.assertEqual(train_result["global_step"], 1.0)


if __name__ == "__main__":
    unittest.main()
