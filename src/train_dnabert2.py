"""Full fine-tuning for DNABERT2 splice-site sequence classification."""

from __future__ import annotations

import argparse
import inspect
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import DatasetDict, load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from .config_utils import ensure_dir, load_config, set_seed, setup_logging, write_json
from .metrics_utils import LABEL_NAMES, classification_metrics, softmax, trainer_scalar_metrics
from .model_utils import copy_remote_model_code, disable_remote_flash_attention


def log_runtime() -> None:
    """Log CUDA and distributed runtime details."""
    logging.info("torch=%s cuda_runtime=%s cuda_available=%s", torch.__version__, torch.version.cuda, torch.cuda.is_available())
    logging.info("cuda_device_count=%s", torch.cuda.device_count())
    for idx in range(torch.cuda.device_count()):
        logging.info("cuda_device_%s=%s", idx, torch.cuda.get_device_name(idx))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        logging.info("distributed world_size=%s rank=%s", torch.distributed.get_world_size(), torch.distributed.get_rank())


def load_dataset(dataset_dir: str | Path, smoke_limit: int | None = None) -> DatasetDict:
    """Load a saved DatasetDict and optionally keep a tiny smoke subset."""
    dataset = load_from_disk(str(dataset_dir))
    if smoke_limit:
        limited = DatasetDict()
        for split, ds in dataset.items():
            limit = min(smoke_limit, len(ds))
            limited[split] = ds.shuffle(seed=42).select(range(limit))
        dataset = limited
    for split, ds in dataset.items():
        labels = pd.Series(ds["label"]).value_counts().sort_index().to_dict()
        logging.info("%s size=%s label_counts=%s", split, len(ds), labels)
    return dataset


def tokenize_dataset(dataset: DatasetDict, tokenizer: Any, token_cfg: dict[str, Any]) -> DatasetDict:
    """Tokenize raw DNA strings for DNABERT2 without manual k-merization."""
    max_length = int(token_cfg.get("max_length", 512))
    padding = token_cfg.get("padding", "max_length")

    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        tokenized = tokenizer(
            batch["sequence"],
            truncation=True,
            padding=padding,
            max_length=max_length,
        )
        tokenized["labels"] = batch["label"]
        return tokenized

    return dataset.map(tokenize_batch, batched=True, desc="Tokenizing DNA sequences")


def training_arguments(config: dict[str, Any], output_dir: Path) -> TrainingArguments:
    """Build TrainingArguments across transformers versions."""
    train_cfg = config.get("training", {})
    sig = inspect.signature(TrainingArguments.__init__)
    strategy_key = "eval_strategy" if "eval_strategy" in sig.parameters else "evaluation_strategy"
    save_strategy_key = "save_strategy"

    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir / "checkpoints"),
        strategy_key: train_cfg.get("eval_strategy", "steps"),
        save_strategy_key: train_cfg.get("save_strategy", "steps"),
        "eval_steps": int(train_cfg.get("eval_steps", 500)),
        "save_steps": int(train_cfg.get("save_steps", 500)),
        "logging_steps": int(train_cfg.get("logging_steps", 50)),
        "learning_rate": float(train_cfg.get("learning_rate", 2e-5)),
        "per_device_train_batch_size": int(train_cfg.get("per_device_train_batch_size", 32)),
        "per_device_eval_batch_size": int(train_cfg.get("per_device_eval_batch_size", 64)),
        "gradient_accumulation_steps": int(train_cfg.get("gradient_accumulation_steps", 1)),
        "num_train_epochs": float(train_cfg.get("num_train_epochs", 3)),
        "max_steps": int(train_cfg.get("max_steps", -1)),
        "weight_decay": float(train_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(train_cfg.get("warmup_ratio", 0.05)),
        "eval_accumulation_steps": int(train_cfg.get("eval_accumulation_steps", 16)),
        "load_best_model_at_end": bool(train_cfg.get("load_best_model_at_end", True)),
        "metric_for_best_model": train_cfg.get("metric_for_best_model", "macro_f1"),
        "greater_is_better": bool(train_cfg.get("greater_is_better", True)),
        "report_to": train_cfg.get("report_to", "none"),
        "save_total_limit": int(train_cfg.get("save_total_limit", 2)),
        "remove_unused_columns": True,
        "dataloader_num_workers": int(train_cfg.get("dataloader_num_workers", 4)),
        "seed": int(config.get("seed", 42)),
    }
    if "bf16" in sig.parameters:
        kwargs["bf16"] = bool(train_cfg.get("bf16", torch.cuda.is_available()))
    if "fp16" in sig.parameters:
        kwargs["fp16"] = bool(train_cfg.get("fp16", False))
    if "optim" in sig.parameters:
        kwargs["optim"] = train_cfg.get("optim", "adamw_torch")
    return TrainingArguments(**kwargs)


def extract_logits(predictions: Any) -> np.ndarray:
    """Handle models that return extra tensors in Trainer prediction output."""
    if isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    return np.asarray(predictions)


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    """Trainer metric callback."""
    logits, labels = eval_pred
    return trainer_scalar_metrics(np.asarray(labels), softmax(extract_logits(logits)))


def preprocess_logits_for_metrics(logits: Any, labels: Any) -> Any:
    """Keep only compact class logits before Trainer caches eval predictions."""
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    return logits


def save_predictions(
    trainer: Trainer,
    tokenized_dataset: DatasetDict,
    raw_dataset: DatasetDict,
    output_dir: Path,
) -> dict[str, Any]:
    """Run test prediction, save probabilities and full metrics."""
    prediction = trainer.predict(tokenized_dataset["test"])
    logits = extract_logits(prediction.predictions)
    labels = np.asarray(prediction.label_ids)
    probs = softmax(logits)
    metrics = classification_metrics(labels, probs)

    frame = raw_dataset["test"].to_pandas()
    for idx, name in enumerate(LABEL_NAMES):
        frame[f"prob_{idx}_{name}"] = probs[:, idx]
    frame["pred_label"] = np.argmax(probs, axis=1)
    frame["splice_site_probability"] = probs[:, 1] + probs[:, 2]
    frame.to_parquet(output_dir / "test_predictions.parquet", index=False)
    write_json(metrics, output_dir / "test_metrics.json")
    return metrics


def train(config: dict[str, Any]) -> dict[str, Any]:
    """Fine-tune DNABERT2 and save artifacts."""
    set_seed(int(config.get("seed", 42)))
    output_dir = ensure_dir(config["output"]["output_dir"])
    log_runtime()

    dataset = load_dataset(config["data"]["dataset_dir"], config.get("smoke", {}).get("max_examples_per_split"))
    model_name = config.get("model", {}).get("name_or_path", "zhihan1996/DNABERT-2-117M")
    trust_remote_code = bool(config.get("model", {}).get("trust_remote_code", True))
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    tokenized = tokenize_dataset(dataset, tokenizer, config.get("tokenization", {}))

    id2label = {idx: name for idx, name in enumerate(LABEL_NAMES)}
    label2id = {name: idx for idx, name in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )
    if bool(config.get("model", {}).get("disable_triton_flash_attention", True)):
        disable_remote_flash_attention(model)

    args = training_arguments(config, output_dir)
    trainer_kwargs = {
        "model": model,
        "args": args,
        "train_dataset": tokenized["train"],
        "eval_dataset": tokenized["valid"],
        "compute_metrics": compute_metrics,
        "preprocess_logits_for_metrics": preprocess_logits_for_metrics,
    }
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()
    final_model_dir = ensure_dir(output_dir / "final_model")
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))
    copy_remote_model_code(model, final_model_dir)
    trainer.save_state()

    valid_metrics = trainer.evaluate(tokenized["valid"])
    test_metrics = save_predictions(trainer, tokenized, dataset, output_dir)
    payload = {
        "train": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in train_result.metrics.items()},
        "valid": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in valid_metrics.items()},
        "test": test_metrics,
    }
    write_json(payload, output_dir / "metrics.json")
    logging.info("Saved model and metrics to %s", output_dir)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to DNABERT2 training YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    train(config)


if __name__ == "__main__":
    main()
