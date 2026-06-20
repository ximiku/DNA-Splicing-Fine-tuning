"""Shared classification metrics for training, baselines, and evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


LABELS = [0, 1, 2]
LABEL_NAMES = ["non_splice", "donor", "acceptor"]


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    logits = np.asarray(logits)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def classification_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    """Compute scalar multiclass metrics plus JSON-friendly per-class details."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs)
    preds = np.argmax(probs, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=LABELS,
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, labels=LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=LABELS).astype(int).tolist(),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(LABELS))
        },
    }

    binary_labels = label_binarize(labels, classes=LABELS)
    for i, name in enumerate(LABEL_NAMES):
        if len(np.unique(binary_labels[:, i])) < 2:
            metrics[f"auroc_{name}"] = None
            metrics[f"auprc_{name}"] = None
            continue
        metrics[f"auroc_{name}"] = float(roc_auc_score(binary_labels[:, i], probs[:, i]))
        metrics[f"auprc_{name}"] = float(average_precision_score(binary_labels[:, i], probs[:, i]))
    return metrics


def trainer_scalar_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    """Flatten selected metrics for HuggingFace Trainer logging."""
    metrics = classification_metrics(labels, probs)
    flat: dict[str, float] = {
        "accuracy": float(metrics["accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
    }
    for class_name, values in metrics["per_class"].items():
        flat[f"f1_{class_name}"] = float(values["f1"])
        flat[f"precision_{class_name}"] = float(values["precision"])
        flat[f"recall_{class_name}"] = float(values["recall"])
    for key, value in metrics.items():
        if key.startswith("auroc_") or key.startswith("auprc_"):
            if value is not None:
                flat[key] = float(value)
    return flat

