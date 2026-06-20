"""Generate visual diagnostics from splice-site prediction tables."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from .config_utils import ensure_dir, write_json
from .evaluate import probability_columns
from .metrics_utils import LABEL_NAMES, LABELS, classification_metrics


DNA_BASES = ["A", "C", "G", "T"]


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a prediction table from parquet, jsonl, or csv."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def save_markdown_table(frame: pd.DataFrame, path: str | Path, index: bool = False) -> None:
    """Write a compact markdown table without adding an extra dependency."""
    path = Path(path)
    ensure_dir(path.parent)
    table = frame.reset_index() if index else frame.copy()
    columns = [str(col) for col in table.columns]

    def format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.6g}"
        text = str(value)
        return text.replace("|", "\\|").replace("\n", " ")

    with path.open("w", encoding="utf-8") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for _, row in table.iterrows():
            handle.write("| " + " | ".join(format_value(row[col]) for col in table.columns) + " |\n")


def plot_confusion(frame: pd.DataFrame, outdir: Path) -> Path:
    """Render the 3-class confusion matrix."""
    matrix = confusion_matrix(frame["label"], frame["pred_label"], labels=LABELS)
    fig, ax = plt.subplots(figsize=(6.8, 5.8), dpi=160)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES)), LABEL_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(len(LABEL_NAMES)), LABEL_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("DNABERT2 Test Confusion Matrix")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = int(matrix[row, col])
            color = "white" if value > matrix.max() * 0.55 else "black"
            ax.text(col, row, f"{value:,}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output = outdir / "confusion_matrix.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def plot_roc_pr(frame: pd.DataFrame, prob_cols: list[str], outdir: Path) -> dict[str, Path]:
    """Render one-vs-rest ROC and precision-recall curves."""
    labels = frame["label"].to_numpy()
    probs = frame[prob_cols].to_numpy()
    binary = label_binarize(labels, classes=LABELS)
    outputs: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=160)
    for idx, name in enumerate(LABEL_NAMES):
        if len(np.unique(binary[:, idx])) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary[:, idx], probs[:, idx])
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} AUROC={auc(fpr, tpr):.4f}")
    ax.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    outputs["roc"] = outdir / "roc_curve.png"
    fig.savefig(outputs["roc"])
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 5.6), dpi=160)
    for idx, name in enumerate(LABEL_NAMES):
        if len(np.unique(binary[:, idx])) < 2:
            continue
        precision, recall, _ = precision_recall_curve(binary[:, idx], probs[:, idx])
        ax.plot(recall, precision, linewidth=2, label=f"{name} AUPRC={auc(recall, precision):.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("One-vs-Rest Precision-Recall Curves")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    outputs["pr"] = outdir / "pr_curve.png"
    fig.savefig(outputs["pr"])
    plt.close(fig)
    return outputs


def plot_probability_distributions(frame: pd.DataFrame, outdir: Path) -> Path:
    """Render splice-site probability distributions by true group."""
    fig, ax = plt.subplots(figsize=(8.4, 5.8), dpi=160)
    bins = np.linspace(0.0, 1.0, 51)
    groups = {
        "donor": frame[frame["label"] == 1]["splice_site_probability"],
        "acceptor": frame[frame["label"] == 2]["splice_site_probability"],
    }
    if "negative_type" in frame.columns:
        for name, group in frame[frame["label"] == 0].groupby("negative_type"):
            if str(name):
                groups[f"negative:{name}"] = group["splice_site_probability"]
    else:
        groups["non_splice"] = frame[frame["label"] == 0]["splice_site_probability"]
    for name, values in groups.items():
        ax.hist(values, bins=bins, histtype="step", linewidth=1.8, density=True, label=name)
    ax.set_yscale("log")
    ax.set_xlabel("P(donor) + P(acceptor)")
    ax.set_ylabel("Density (log scale)")
    ax.set_title("Splice-Site Probability Distributions")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    output = outdir / "splice_probability_distribution.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def centered_base_frequencies(sequences: pd.Series, flank: int) -> np.ndarray:
    """Return base frequencies around the center of equal-length DNA windows."""
    if sequences.empty:
        return np.zeros((len(DNA_BASES), 2 * flank + 1), dtype=float)
    length = len(sequences.iloc[0])
    center = length // 2
    start = max(0, center - flank)
    end = min(length, center + flank + 1)
    window_len = end - start
    counts = np.zeros((len(DNA_BASES), window_len), dtype=float)
    base_to_idx = {base: idx for idx, base in enumerate(DNA_BASES)}
    for seq in sequences.astype(str).str.upper():
        subseq = seq[start:end]
        for pos, base in enumerate(subseq):
            idx = base_to_idx.get(base)
            if idx is not None:
                counts[idx, pos] += 1
    col_sums = counts.sum(axis=0, keepdims=True)
    return np.divide(counts, col_sums, out=np.zeros_like(counts), where=col_sums > 0)


def plot_sequence_frequencies(frame: pd.DataFrame, outdir: Path, sample_per_label: int = 5000, flank: int = 20) -> Path:
    """Render centered nucleotide frequency heatmaps for each class."""
    rng = np.random.default_rng(42)
    class_frames = []
    for label, name in enumerate(LABEL_NAMES):
        group = frame[frame["label"] == label]
        if len(group) > sample_per_label:
            group = group.iloc[rng.choice(len(group), size=sample_per_label, replace=False)]
        class_frames.append((name, group))

    fig, axes = plt.subplots(len(class_frames), 1, figsize=(10.5, 5.8), dpi=160, sharex=True)
    if len(class_frames) == 1:
        axes = [axes]
    positions = np.arange(-flank, flank + 1)
    for ax, (name, group) in zip(axes, class_frames, strict=True):
        freqs = centered_base_frequencies(group["sequence"], flank)
        image = ax.imshow(freqs, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=1)
        ax.set_yticks(range(len(DNA_BASES)), DNA_BASES)
        ax.set_ylabel(name)
        ax.axvline(flank, color="white", linewidth=0.8, alpha=0.7)
    axes[-1].set_xticks(range(0, len(positions), 5), positions[::5])
    axes[-1].set_xlabel("Position relative to centered candidate site")
    fig.suptitle("Centered Nucleotide Frequency Patterns")
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Frequency")
    output = outdir / "centered_sequence_frequency.png"
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    return output


def negative_type_tables(frame: pd.DataFrame, outdir: Path) -> dict[str, Path]:
    """Save negative-type summary and high-scoring motif-hard false positives."""
    outputs: dict[str, Path] = {}
    if "negative_type" not in frame.columns:
        return outputs

    negative = frame[frame["label"] == 0].copy()
    if negative.empty:
        return outputs
    negative["is_false_positive"] = negative["pred_label"] != 0
    grouped = []
    for name, group in negative.groupby("negative_type"):
        if not str(name):
            continue
        grouped.append(
            {
                "negative_type": str(name),
                "n": int(len(group)),
                "accuracy": float((group["pred_label"] == 0).mean()),
                "false_positive_rate": float((group["pred_label"] != 0).mean()),
                "donor_fp": int((group["pred_label"] == 1).sum()),
                "acceptor_fp": int((group["pred_label"] == 2).sum()),
                "mean_splice_site_probability": float(group["splice_site_probability"].mean()),
                "median_splice_site_probability": float(group["splice_site_probability"].median()),
                "max_splice_site_probability": float(group["splice_site_probability"].max()),
            }
        )
    summary = pd.DataFrame(grouped).sort_values("false_positive_rate", ascending=False)
    outputs["negative_summary_csv"] = outdir / "negative_type_summary.csv"
    outputs["negative_summary_md"] = outdir / "negative_type_summary.md"
    summary.to_csv(outputs["negative_summary_csv"], index=False)
    save_markdown_table(summary, outputs["negative_summary_md"])

    motif_fp = negative[(negative["negative_type"] == "motif_hard") & (negative["pred_label"] != 0)].copy()
    if not motif_fp.empty:
        cols = [
            "chrom",
            "pos",
            "strand",
            "pred_label",
            "prob_0_non_splice",
            "prob_1_donor",
            "prob_2_acceptor",
            "splice_site_probability",
            "sequence",
        ]
        motif_fp = motif_fp.sort_values("splice_site_probability", ascending=False)
        outputs["motif_fp_csv"] = outdir / "motif_hard_false_positives_top100.csv"
        outputs["motif_fp_md"] = outdir / "motif_hard_false_positives_top20.md"
        motif_fp[cols].head(100).to_csv(outputs["motif_fp_csv"], index=False)
        display = motif_fp[cols].head(20).copy()
        display["sequence"] = display["sequence"].str.slice(180, 221)
        save_markdown_table(display, outputs["motif_fp_md"])
    return outputs


def plot_training_curves(trainer_state: str | Path | None, outdir: Path) -> Path | None:
    """Render train loss and validation macro-F1 from a Trainer state file."""
    if trainer_state is None:
        return None
    path = Path(trainer_state)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    history = state.get("log_history", [])
    train_rows = [row for row in history if "loss" in row and "step" in row]
    eval_rows = [row for row in history if "eval_macro_f1" in row and "step" in row]
    if not train_rows and not eval_rows:
        return None

    fig, ax1 = plt.subplots(figsize=(8.2, 5.4), dpi=160)
    if train_rows:
        ax1.plot([row["step"] for row in train_rows], [row["loss"] for row in train_rows], color="#2868a8", label="train loss")
        ax1.set_ylabel("Train loss", color="#2868a8")
        ax1.tick_params(axis="y", labelcolor="#2868a8")
    ax1.set_xlabel("Step")
    ax1.grid(alpha=0.25)
    if eval_rows:
        ax2 = ax1.twinx()
        ax2.plot(
            [row["step"] for row in eval_rows],
            [row["eval_macro_f1"] for row in eval_rows],
            color="#b4464b",
            marker="o",
            markersize=3,
            label="valid macro-F1",
        )
        ax2.set_ylabel("Validation macro-F1", color="#b4464b")
        ax2.tick_params(axis="y", labelcolor="#b4464b")
    fig.suptitle("Training Loss and Validation Macro-F1")
    fig.tight_layout()
    output = outdir / "training_curves.png"
    fig.savefig(output)
    plt.close(fig)
    return output


def generate_visualizations(
    predictions: Path,
    outdir: Path,
    trainer_state: Path | None = None,
    title: str | None = None,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Generate a standard visualization bundle from predictions."""
    if outdir.exists() and (outdir / "visualization_manifest.json").exists() and not allow_overwrite:
        base = outdir
        suffix = 2
        while outdir.exists():
            outdir = base.with_name(f"{base.name}_v{suffix}")
            suffix += 1
    outdir = ensure_dir(outdir)
    frame = read_table(predictions)
    prob_cols = probability_columns(frame)
    if "pred_label" not in frame.columns:
        frame["pred_label"] = frame[prob_cols].to_numpy().argmax(axis=1)
    if "splice_site_probability" not in frame.columns:
        frame["splice_site_probability"] = frame[prob_cols[1]] + frame[prob_cols[2]]

    metrics = classification_metrics(frame["label"].to_numpy(), frame[prob_cols].to_numpy())
    write_json(metrics, outdir / "metrics_from_predictions.json")

    outputs: dict[str, Any] = {
        "title": title or predictions.stem,
        "predictions": str(predictions),
        "outdir": str(outdir),
        "metrics": str(outdir / "metrics_from_predictions.json"),
        "figures": {},
        "tables": {},
    }
    outputs["figures"]["confusion_matrix"] = str(plot_confusion(frame, outdir))
    outputs["figures"].update({key: str(value) for key, value in plot_roc_pr(frame, prob_cols, outdir).items()})
    outputs["figures"]["splice_probability_distribution"] = str(plot_probability_distributions(frame, outdir))
    if "sequence" in frame.columns:
        outputs["figures"]["centered_sequence_frequency"] = str(plot_sequence_frequencies(frame, outdir))
    training_curve = plot_training_curves(trainer_state, outdir)
    if training_curve is not None:
        outputs["figures"]["training_curves"] = str(training_curve)
    outputs["tables"].update({key: str(value) for key, value in negative_type_tables(frame, outdir).items()})
    write_json(outputs, outdir / "visualization_manifest.json")
    logging.info("Saved visualization bundle to %s", outdir)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Prediction parquet/csv/jsonl with labels and probability columns")
    parser.add_argument("--output_dir", required=True, help="Directory for figures, tables, and manifest")
    parser.add_argument("--trainer_state", default=None, help="Optional HuggingFace trainer_state.json")
    parser.add_argument("--title", default=None, help="Human-readable experiment title for manifest metadata")
    parser.add_argument("--allow_overwrite", action="store_true", help="Write into an existing visualization bundle directory")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    generate_visualizations(
        predictions=Path(args.predictions),
        outdir=Path(args.output_dir),
        trainer_state=Path(args.trainer_state) if args.trainer_state else None,
        title=args.title,
        allow_overwrite=bool(args.allow_overwrite),
    )


if __name__ == "__main__":
    main()
