import inspect
import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Optional, Sequence

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten


def default_data_collator(features: list[Mapping[str, Any]]) -> dict[str, mx.array]:
    if not features:
        raise ValueError("Cannot collate an empty batch.")

    batch: dict[str, mx.array] = {}
    for key in features[0]:
        values = [feature[key] for feature in features]
        first = values[0]

        if isinstance(first, mx.array):
            batch[key] = mx.stack(values)
            continue

        if isinstance(first, np.ndarray):
            batch[key] = mx.array(np.stack(values))
            continue

        batch[key] = mx.array(values)

    return batch


@dataclass
class EvalPrediction:
    predictions: Any
    label_ids: Any


@dataclass
class TrainingArguments:
    output_dir: str = "trainer_output"
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    num_train_epochs: int = 3
    max_steps: int = -1
    gradient_accumulation_steps: int = 1
    logging_steps: int = 50
    eval_steps: Optional[int] = None
    save_steps: Optional[int] = None
    max_grad_norm: Optional[float] = None
    seed: Optional[int] = None


@dataclass
class TrainerState:
    epoch: float = 0.0
    global_step: int = 0
    train_loss: float = 0.0
    best_eval_loss: Optional[float] = None
    log_history: list[dict[str, Any]] = field(default_factory=list)


class TrainerCallback:
    def on_train_begin(self, args: TrainingArguments, state: TrainerState) -> None:
        pass

    def on_log(
        self, args: TrainingArguments, state: TrainerState, logs: dict[str, Any]
    ) -> None:
        pass

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, metrics: dict[str, Any]
    ) -> None:
        pass

    def on_save(
        self, args: TrainingArguments, state: TrainerState, checkpoint_dir: Path
    ) -> None:
        pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState) -> None:
        pass


class Trainer:
    _LABEL_KEYS = ("labels", "label", "label_ids", "start_positions", "end_positions")

    def __init__(
        self,
        model: nn.Module,
        args: Optional[TrainingArguments] = None,
        train_dataset: Optional[
            Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]
        ] = None,
        eval_dataset: Optional[
            Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]
        ] = None,
        data_collator: Optional[
            Callable[[list[Mapping[str, Any]]], dict[str, mx.array]]
        ] = None,
        optimizers: Optional[Any] = None,
        compute_loss_func: Optional[
            Callable[[nn.Module, Mapping[str, Any], Any], mx.array]
        ] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], dict[str, float]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
    ) -> None:
        self.model = model
        self.args = args or TrainingArguments()
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator or default_data_collator
        self.optimizer = optimizers
        self.compute_loss_func = compute_loss_func
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks or []
        self.state = TrainerState()
        self._model_forward_arg_names, self._model_accepts_kwargs = (
            self._inspect_model_forward_signature()
        )
        self._loss_and_grad = nn.value_and_grad(self.model, self._loss_fn)

        if self.args.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be >= 1.")
        if self.args.per_device_train_batch_size < 1:
            raise ValueError("per_device_train_batch_size must be >= 1.")
        if self.args.per_device_eval_batch_size < 1:
            raise ValueError("per_device_eval_batch_size must be >= 1.")

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        if self.args.weight_decay > 0:
            self.optimizer = optim.AdamW(
                learning_rate=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(learning_rate=self.args.learning_rate)
        return self.optimizer

    def _loss_fn(self, model: nn.Module, inputs: Mapping[str, Any]):
        return self.compute_loss(model, inputs, return_outputs=False)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Mapping[str, Any],
        return_outputs: bool = False,
    ):
        outputs = model(**self._prepare_model_inputs(inputs))
        loss = (
            self.compute_loss_func(model, inputs, outputs)
            if self.compute_loss_func is not None
            else self._extract_loss(outputs)
        )
        loss = self._ensure_scalar_loss(loss)
        return (loss, outputs) if return_outputs else loss

    def train(self) -> dict[str, float]:
        if self.train_dataset is None:
            raise ValueError("Trainer.train() requires a train_dataset.")

        if self.args.seed is not None:
            np.random.seed(self.args.seed)
            if hasattr(mx, "random") and hasattr(mx.random, "seed"):
                mx.random.seed(self.args.seed)

        self.create_optimizer()
        self.model.train()
        mx.eval(self.model.parameters())

        for callback in self.callbacks:
            callback.on_train_begin(self.args, self.state)

        total_loss = 0.0
        updates_since_log = 0
        logged_loss_total = 0.0
        max_steps = self.args.max_steps

        for epoch_index in range(self.args.num_train_epochs):
            accumulated_grads = None
            accumulated_loss = 0.0
            accumulated_batches = 0

            for step_in_epoch, batch in enumerate(self.get_train_dataloader(), start=1):
                loss, grads = self._loss_and_grad(self.model, batch)
                mx.eval(loss)

                accumulated_grads = self._add_trees(accumulated_grads, grads)
                accumulated_loss += float(loss.item())
                accumulated_batches += 1

                should_update = (
                    accumulated_batches == self.args.gradient_accumulation_steps
                )
                if should_update:
                    step_loss = accumulated_loss / accumulated_batches
                    self._apply_gradients(accumulated_grads, accumulated_batches)
                    accumulated_grads = None
                    accumulated_loss = 0.0
                    accumulated_batches = 0
                    total_loss += step_loss
                    logged_loss_total += step_loss
                    updates_since_log += 1
                    self.state.epoch = epoch_index + (
                        step_in_epoch / max(self._num_train_batches(), 1)
                    )
                    self.state.global_step += 1
                    updates_since_log, logged_loss_total = self._post_update(
                        updates_since_log,
                        logged_loss_total,
                    )

                    if max_steps > 0 and self.state.global_step >= max_steps:
                        break

            if accumulated_batches:
                step_loss = accumulated_loss / accumulated_batches
                self._apply_gradients(accumulated_grads, accumulated_batches)
                total_loss += step_loss
                logged_loss_total += step_loss
                updates_since_log += 1
                self.state.global_step += 1
                self.state.epoch = float(epoch_index + 1)
                updates_since_log, logged_loss_total = self._post_update(
                    updates_since_log,
                    logged_loss_total,
                )

            if max_steps > 0 and self.state.global_step >= max_steps:
                break

        self.state.train_loss = total_loss / max(self.state.global_step, 1)

        if updates_since_log:
            self._log(
                {
                    "loss": logged_loss_total / max(updates_since_log, 1),
                    "epoch": self.state.epoch,
                    "step": self.state.global_step,
                }
            )

        self.save_model()
        for callback in self.callbacks:
            callback.on_train_end(self.args, self.state)

        return {
            "global_step": float(self.state.global_step),
            "train_loss": self.state.train_loss,
        }

    def evaluate(
        self,
        eval_dataset: Optional[
            Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]
        ] = None,
    ) -> dict[str, Any]:
        dataset = eval_dataset or self.eval_dataset
        if dataset is None:
            raise ValueError("Trainer.evaluate() requires an eval_dataset.")

        was_training = getattr(self.model, "training", False)
        self.model.eval()

        losses: list[float] = []
        predictions: list[Any] = []
        labels: list[Any] = []

        for batch in self.get_eval_dataloader(dataset):
            loss, outputs = self.compute_loss(self.model, batch, return_outputs=True)
            if loss is not None:
                mx.eval(loss)
                losses.append(float(loss.item()))

            if self.compute_metrics is not None:
                batch_predictions = self._extract_predictions(outputs)
                if batch_predictions is None:
                    raise ValueError(
                        "Model outputs do not expose predictions for compute_metrics."
                    )
                predictions.append(self._to_numpy(batch_predictions))
                labels.append(self._to_numpy(self._extract_labels(batch)))

        metrics: dict[str, Any] = {}
        if losses:
            eval_loss = float(np.mean(losses))
            metrics["eval_loss"] = eval_loss
            if (
                self.state.best_eval_loss is None
                or eval_loss < self.state.best_eval_loss
            ):
                self.state.best_eval_loss = eval_loss

        if self.compute_metrics is not None and predictions:
            metrics.update(
                self.compute_metrics(
                    EvalPrediction(
                        predictions=self._stack_nested(predictions),
                        label_ids=self._stack_nested(labels),
                    )
                )
            )

        self._log({"step": self.state.global_step, **metrics})
        for callback in self.callbacks:
            callback.on_evaluate(self.args, self.state, metrics)

        if was_training:
            self.model.train()

        return metrics

    def save_model(self, output_dir: Optional[str] = None) -> Path:
        checkpoint_dir = Path(output_dir or self.args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        mx.save_safetensors(
            str(checkpoint_dir / "model.safetensors"),
            dict(tree_flatten(self.model.parameters())),
        )
        if self.optimizer is not None:
            mx.save_safetensors(
                str(checkpoint_dir / "optimizer.safetensors"),
                dict(tree_flatten(self.optimizer.state)),
            )

        with (checkpoint_dir / "training_args.json").open("w", encoding="utf-8") as f:
            json.dump(self._json_ready(asdict(self.args)), f, indent=2)
        with (checkpoint_dir / "trainer_state.json").open("w", encoding="utf-8") as f:
            json.dump(self._json_ready(asdict(self.state)), f, indent=2)

        return checkpoint_dir

    def get_train_dataloader(self) -> Iterator[dict[str, mx.array]]:
        return self._iter_dataset(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
        )

    def get_eval_dataloader(
        self,
        eval_dataset: Optional[
            Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]
        ] = None,
    ) -> Iterator[dict[str, mx.array]]:
        return self._iter_dataset(
            eval_dataset or self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
        )

    def _iter_dataset(
        self,
        dataset: Optional[Sequence[Mapping[str, Any]] | Iterable[Mapping[str, Any]]],
        batch_size: int,
        shuffle: bool,
    ) -> Iterator[dict[str, mx.array]]:
        if dataset is None:
            return

        if self._is_indexable_dataset(dataset):
            indices = np.arange(len(dataset))
            if shuffle:
                np.random.shuffle(indices)

            for start in range(0, len(indices), batch_size):
                batch_examples = [
                    dataset[idx] for idx in indices[start : start + batch_size]
                ]
                yield self.data_collator(batch_examples)
            return

        if shuffle:
            raise ValueError("Shuffling iterable datasets is not supported.")

        batch_examples: list[Mapping[str, Any]] = []
        for example in dataset:
            batch_examples.append(example)
            if len(batch_examples) == batch_size:
                yield self.data_collator(batch_examples)
                batch_examples = []

        if batch_examples:
            yield self.data_collator(batch_examples)

    def _num_train_batches(self) -> int:
        if self.train_dataset is None:
            return 0
        if not self._is_indexable_dataset(self.train_dataset):
            if self.args.max_steps <= 0:
                raise ValueError("Iterable train datasets require max_steps to be set.")
            return self.args.max_steps

        return math.ceil(
            len(self.train_dataset) / self.args.per_device_train_batch_size
        )

    def _apply_gradients(self, grads: Any, batches_in_step: int) -> None:
        grads = self._scale_tree(grads, 1.0 / batches_in_step)
        if self.args.max_grad_norm is not None:
            grads, _ = optim.clip_grad_norm(grads, self.args.max_grad_norm)

        self.optimizer.update(self.model, grads)
        mx.eval(self.model.parameters(), self.optimizer.state)

    def _save_checkpoint(self) -> None:
        checkpoint_dir = self.save_model(
            str(Path(self.args.output_dir) / f"checkpoint-{self.state.global_step}")
        )
        for callback in self.callbacks:
            callback.on_save(self.args, self.state, checkpoint_dir)

    def _post_update(
        self,
        updates_since_log: int,
        logged_loss_total: float,
    ) -> tuple[int, float]:
        if (
            self.args.logging_steps
            and self.state.global_step % self.args.logging_steps == 0
        ):
            self._log(
                {
                    "loss": logged_loss_total / max(updates_since_log, 1),
                    "epoch": self.state.epoch,
                    "step": self.state.global_step,
                }
            )
            updates_since_log = 0
            logged_loss_total = 0.0

        if (
            self.args.eval_steps
            and self.eval_dataset is not None
            and self.state.global_step % self.args.eval_steps == 0
        ):
            self.evaluate()

        if self.args.save_steps and self.state.global_step % self.args.save_steps == 0:
            self._save_checkpoint()

        return updates_since_log, logged_loss_total

    def _log(self, logs: dict[str, Any]) -> None:
        logs = self._json_ready(logs)
        self.state.log_history.append(logs)
        for callback in self.callbacks:
            callback.on_log(self.args, self.state, logs)

    @staticmethod
    def _extract_loss(outputs: Any, allow_missing: bool = False):
        if hasattr(outputs, "loss"):
            return outputs.loss
        if isinstance(outputs, (tuple, list)) and outputs:
            return outputs[0]
        if allow_missing:
            return None
        raise ValueError("Model outputs do not expose a loss.")

    @classmethod
    def _extract_labels(cls, batch: Mapping[str, Any]) -> Any:
        label_keys = [key for key in cls._LABEL_KEYS if key in batch]
        if not label_keys:
            return None
        if len(label_keys) == 1:
            return batch[label_keys[0]]
        return {key: batch[key] for key in label_keys}

    @staticmethod
    def _extract_predictions(outputs: Any) -> Any:
        if hasattr(outputs, "logits"):
            return outputs.logits
        if hasattr(outputs, "start_logits") and hasattr(outputs, "end_logits"):
            return {
                "start_logits": outputs.start_logits,
                "end_logits": outputs.end_logits,
            }
        if isinstance(outputs, (tuple, list)):
            if len(outputs) > 1:
                return outputs[1]
            return None
        return outputs

    @staticmethod
    def _to_numpy(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, mx.array):
            return np.array(value)
        if isinstance(value, Mapping):
            return {key: Trainer._to_numpy(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [Trainer._to_numpy(item) for item in value]
        return np.array(value)

    @staticmethod
    def _stack_nested(values: list[Any]) -> Any:
        first = values[0]
        if first is None:
            return None
        if isinstance(first, dict):
            return {
                key: Trainer._stack_nested([value[key] for value in values])
                for key in first
            }
        if isinstance(first, list):
            return [
                Trainer._stack_nested([value[idx] for value in values])
                for idx in range(len(first))
            ]
        try:
            return np.concatenate(values, axis=0)
        except ValueError:
            warnings.warn(
                "Could not concatenate evaluation outputs; returning a list instead.",
                stacklevel=2,
            )
            return values

    @staticmethod
    def _scale_tree(tree: Any, scale: float) -> Any:
        flattened = tree_flatten(tree)
        return tree_unflatten([(key, value * scale) for key, value in flattened])

    @staticmethod
    def _add_trees(left: Any, right: Any) -> Any:
        if left is None:
            return right

        left_flat = dict(tree_flatten(left))
        right_flat = dict(tree_flatten(right))
        return tree_unflatten(
            [(key, left_flat[key] + right_flat[key]) for key in left_flat]
        )

    @staticmethod
    def _json_ready(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: Trainer._json_ready(item) for key, item in value.items()}
        if isinstance(value, list):
            return [Trainer._json_ready(item) for item in value]
        if isinstance(value, tuple):
            return [Trainer._json_ready(item) for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @staticmethod
    def _is_indexable_dataset(dataset: Any) -> bool:
        return hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__")

    @staticmethod
    def _ensure_scalar_loss(loss: Any):
        if loss is None:
            return None
        if isinstance(loss, (int, float, np.number)):
            return mx.array(loss)
        if getattr(loss, "shape", ()) != ():
            return mx.mean(loss)
        return loss

    def _prepare_model_inputs(self, inputs: Mapping[str, Any]) -> dict[str, Any]:
        model_inputs = dict(inputs)

        if self.compute_loss_func is not None:
            model_inputs = {
                key: value
                for key, value in model_inputs.items()
                if key not in self._LABEL_KEYS
            }

        if self._model_accepts_kwargs:
            return model_inputs

        return {
            key: value
            for key, value in model_inputs.items()
            if key in self._model_forward_arg_names
        }

    def _inspect_model_forward_signature(self) -> tuple[set[str], bool]:
        try:
            signature = inspect.signature(self.model.__call__)
        except (TypeError, ValueError):
            return set(), True

        arg_names: set[str] = set()
        accepts_kwargs = False
        for name, parameter in signature.parameters.items():
            if name == "self":
                continue
            if parameter.kind == inspect.Parameter.VAR_KEYWORD:
                accepts_kwargs = True
                continue
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                arg_names.add(name)

        return arg_names, accepts_kwargs
