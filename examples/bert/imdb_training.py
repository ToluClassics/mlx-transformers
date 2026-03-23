import argparse
import json

import numpy as np
from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from mlx_transformers import Trainer, TrainerCallback, TrainingArguments
from mlx_transformers.models import BertForSequenceClassification


class PrintCallback(TrainerCallback):
    def on_log(self, args, state, logs):
        print(json.dumps(logs, sort_keys=True))

    def on_save(self, args, state, checkpoint_dir):
        print(f"Saved checkpoint to {checkpoint_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a small BERT model on IMDB with the MLX trainer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Hugging Face model checkpoint to use as the BERT backbone.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/imdb-bert",
        help="Directory where checkpoints and trainer state are saved.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=2000,
        help="Number of training examples to use from IMDB.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=500,
        help="Number of evaluation examples to use from IMDB.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length used for tokenization.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=8,
        help="Training batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=16,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--num-train-epochs",
        type=int,
        default=1,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Optional cap on optimizer steps. Use -1 to train for all epochs.",
    )
    parser.add_argument(
        "--logging-steps",
        type=int,
        default=25,
        help="How often to emit training logs.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=100,
        help="How often to run evaluation during training.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=100,
        help="How often to write checkpoints during training.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=1,
        help="Number of batches to accumulate before an optimizer step.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset sampling and training.",
    )
    return parser.parse_args()


def build_dataset(tokenizer, max_length, train_samples, eval_samples, seed):
    dataset = load_dataset("imdb")

    train_count = min(train_samples, len(dataset["train"]))
    eval_count = min(eval_samples, len(dataset["test"]))

    train_dataset = dataset["train"].shuffle(seed=seed).select(range(train_count))
    eval_dataset = dataset["test"].shuffle(seed=seed).select(range(eval_count))

    def tokenize_batch(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    train_dataset = train_dataset.map(tokenize_batch, batched=True)
    eval_dataset = eval_dataset.map(tokenize_batch, batched=True)

    train_dataset = train_dataset.remove_columns(["text"]).rename_column(
        "label", "labels"
    )
    eval_dataset = eval_dataset.remove_columns(["text"]).rename_column(
        "label", "labels"
    )

    return train_dataset, eval_dataset


def compute_metrics(prediction):
    predicted_labels = np.argmax(prediction.predictions, axis=-1)
    accuracy = float((predicted_labels == prediction.label_ids).mean())
    return {"accuracy": accuracy}


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)

    model = BertForSequenceClassification(config)
    model.from_pretrained(args.model_name)

    train_dataset, eval_dataset = build_dataset(
        tokenizer=tokenizer,
        max_length=args.max_length,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        callbacks=[PrintCallback()],
    )

    train_metrics = trainer.train()
    eval_metrics = trainer.evaluate()

    print("Train metrics:")
    print(json.dumps(train_metrics, indent=2, sort_keys=True))
    print("Eval metrics:")
    print(json.dumps(eval_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
