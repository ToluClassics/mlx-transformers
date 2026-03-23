import argparse
import json

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer

from mlx_transformers import LoraConfig, SFTConfig, SFTTrainer, TrainerCallback
from mlx_transformers.models import Qwen3ForCausalLM


class PrintCallback(TrainerCallback):
    def on_log(self, args, state, logs):
        print(json.dumps(logs, sort_keys=True))

    def on_save(self, args, state, checkpoint_dir):
        print(f"Saved checkpoint to {checkpoint_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen/Qwen3-0.6B on a small WildChat subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-name",
        default="Qwen/Qwen3-0.6B",
        help="Hugging Face model checkpoint to fine-tune.",
    )
    parser.add_argument(
        "--dataset-name",
        default="allenai/WildChat",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/qwen3-wildchat-lora-sft",
        help="Directory where checkpoints and trainer state are saved.",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=1000,
        help="Number of WildChat conversations to use for training.",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=200,
        help="Number of WildChat conversations to use for evaluation.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenized sequence length.",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        type=int,
        default=1,
        help="Training batch size.",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=1,
        help="Evaluation batch size.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Number of batches to accumulate before an optimizer step.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-5,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay.",
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
        default=1,
        help="How often to emit training logs.",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=50,
        help="How often to run evaluation during training.",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=50,
        help="How often to write checkpoints during training.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset sampling and training.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--lora-alpha",
        type=float,
        default=16.0,
        help="LoRA alpha.",
    )
    parser.add_argument(
        "--lora-dropout",
        type=float,
        default=0.0,
        help="LoRA dropout.",
    )
    parser.add_argument(
        "--lora-scale",
        type=float,
        default=10.0,
        help="Extra scaling factor applied to the LoRA update.",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of decoder layers, counting from the end, to convert to LoRA.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        help="Optional explicit LoRA target module names such as q_proj v_proj.",
    )
    return parser.parse_args()


def load_wildchat_subset(dataset_name, train_samples, eval_samples, seed):
    total_samples = train_samples + eval_samples
    dataset = load_dataset(dataset_name, split=f"train[:{total_samples}]")

    def has_valid_messages(example):
        valid_messages = 0
        for message in example["conversation"]:
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role in {"system", "user", "assistant"} and content:
                valid_messages += 1
        return valid_messages >= 2

    dataset = dataset.filter(has_valid_messages)
    dataset = dataset.shuffle(seed=seed)

    available = len(dataset)
    train_count = min(train_samples, available)
    eval_count = min(eval_samples, max(available - train_count, 0))

    if train_count == 0:
        raise ValueError("WildChat subset did not contain any valid conversations.")

    train_dataset = dataset.select(range(train_count))
    eval_dataset = dataset.select(range(train_count, train_count + eval_count))

    return train_dataset, eval_dataset


def build_formatting_func(tokenizer):
    def formatting_func(example):
        messages = []
        for message in example["conversation"]:
            role = message.get("role")
            content = str(message.get("content", "")).strip()
            if role not in {"system", "user", "assistant"} or not content:
                continue
            messages.append({"role": role, "content": content})

        if not messages:
            return ""

        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

    return formatting_func


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    config = AutoConfig.from_pretrained(args.model_name)

    model = Qwen3ForCausalLM(config)
    model.from_pretrained(args.model_name)

    train_dataset, eval_dataset = load_wildchat_subset(
        dataset_name=args.dataset_name,
        train_samples=args.train_samples,
        eval_samples=args.eval_samples,
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=SFTConfig(
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
            max_length=args.max_length,
        ),
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=build_formatting_func(tokenizer),
        lora_config=LoraConfig(
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            scale=args.lora_scale,
            num_layers=args.lora_layers,
            target_modules=args.lora_target_modules,
        ),
        callbacks=[PrintCallback()],
    )

    train_metrics = trainer.train()
    print("Train metrics:")
    print(json.dumps(train_metrics, indent=2, sort_keys=True))

    if len(eval_dataset) > 0:
        eval_metrics = trainer.evaluate()
        print("Eval metrics:")
        print(json.dumps(eval_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
