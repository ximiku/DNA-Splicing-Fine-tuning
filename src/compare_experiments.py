"""Build a compact comparison table across DNABERT2 splice-site experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config_utils import ensure_dir


DEFAULT_EXPERIMENTS = [
    {
        "name": "full_baseline",
        "split_or_eval": "random full test",
        "training_mode": "full",
        "trainable_parameters": 117070851,
        "trainable_fraction": 1.0,
        "metrics": "outputs/dnabert2_full/test_metrics.json",
        "run_metrics": "outputs/dnabert2_full/metrics.json",
        "negative_summary": "outputs/visualizations/dnabert2_full_baseline/negative_type_summary.csv",
        "output_dir": "outputs/dnabert2_full",
    },
    {
        "name": "chrom_holdout_full",
        "split_or_eval": "chr9/chr10 test",
        "training_mode": "full",
        "trainable_parameters": 117070851,
        "trainable_fraction": 1.0,
        "metrics": "outputs/dnabert2_chrom_holdout_full/test_metrics.json",
        "run_metrics": "outputs/dnabert2_chrom_holdout_full/metrics.json",
        "negative_summary": "outputs/visualizations/dnabert2_chrom_holdout_full/negative_type_summary.csv",
        "output_dir": "outputs/dnabert2_chrom_holdout_full",
    },
    {
        "name": "random_only_ablation",
        "split_or_eval": "original full test",
        "training_mode": "full",
        "trainable_parameters": 117070851,
        "trainable_fraction": 1.0,
        "metrics": "outputs/dnabert2_ablation_random_only/full_test_metrics.json",
        "run_metrics": "outputs/dnabert2_ablation_random_only/metrics.json",
        "negative_summary": "outputs/visualizations/dnabert2_ablation_random_only/negative_type_summary.csv",
        "output_dir": "outputs/dnabert2_ablation_random_only",
    },
    {
        "name": "linear_probe",
        "split_or_eval": "random full test",
        "metrics": "outputs/dnabert2_linear_probe/test_metrics.json",
        "run_metrics": "outputs/dnabert2_linear_probe/metrics.json",
        "negative_summary": "outputs/visualizations/dnabert2_linear_probe/negative_type_summary.csv",
        "output_dir": "outputs/dnabert2_linear_probe",
    },
    {
        "name": "lora",
        "split_or_eval": "random full test",
        "metrics": "outputs/dnabert2_lora/test_metrics.json",
        "run_metrics": "outputs/dnabert2_lora/metrics.json",
        "negative_summary": "outputs/visualizations/dnabert2_lora/negative_type_summary.csv",
        "output_dir": "outputs/dnabert2_lora",
    },
]


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def directory_size_bytes(path: str | Path) -> int:
    root = Path(path)
    if not root.exists():
        return 0
    if root.is_file():
        return root.stat().st_size
    return sum(item.stat().st_size for item in root.rglob("*") if item.is_file())


def mean_available(metrics: dict[str, Any], names: list[str]) -> float | None:
    values = [float(metrics[name]) for name in names if name in metrics and metrics[name] is not None]
    return float(sum(values) / len(values)) if values else None


def motif_hard_fpr(path: str | Path) -> float | None:
    table_path = Path(path)
    if not table_path.exists():
        return None
    frame = pd.read_csv(table_path)
    if "negative_type" not in frame.columns or "false_positive_rate" not in frame.columns:
        return None
    rows = frame.loc[frame["negative_type"] == "motif_hard", "false_positive_rate"]
    return float(rows.iloc[0]) if len(rows) else None


def max_gpu_memory_gib(run_metrics: dict[str, Any]) -> float | None:
    memory = run_metrics.get("run", {}).get("gpu_memory", {})
    peaks = [device.get("max_reserved_bytes", 0) for device in memory.values() if isinstance(device, dict)]
    if not peaks:
        return None
    return max(peaks) / (1024**3)


def row_from_experiment(spec: dict[str, str]) -> dict[str, Any] | None:
    metrics_path = Path(spec["metrics"])
    run_path = Path(spec["run_metrics"])
    if not metrics_path.exists() or not run_path.exists():
        return None

    metrics = read_json(metrics_path)
    run_metrics = read_json(run_path)
    run_summary = run_metrics.get("run", {})
    artifacts = run_summary.get("artifacts", {})
    output_dir = Path(spec["output_dir"])
    final_model_size = artifacts.get("final_model_size_bytes")
    checkpoint_size = artifacts.get("checkpoints_size_bytes")
    if final_model_size is None:
        final_model_size = directory_size_bytes(output_dir / "final_model")
    if checkpoint_size is None:
        checkpoint_size = directory_size_bytes(output_dir / "checkpoints")

    return {
        "experiment": spec["name"],
        "split_or_eval": spec["split_or_eval"],
        "training_mode": run_summary.get("training_mode", spec.get("training_mode", "full")),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "mean_auroc": mean_available(metrics, ["auroc_non_splice", "auroc_donor", "auroc_acceptor"]),
        "mean_auprc": mean_available(metrics, ["auprc_non_splice", "auprc_donor", "auprc_acceptor"]),
        "motif_hard_fpr": motif_hard_fpr(spec["negative_summary"]),
        "train_runtime_sec": run_summary.get(
            "wall_clock_train_runtime_sec",
            run_metrics.get("train", {}).get("train_runtime"),
        ),
        "peak_gpu_reserved_gib": max_gpu_memory_gib(run_metrics),
        "trainable_parameters": run_summary.get("parameters", {}).get(
            "trainable_parameters", spec.get("trainable_parameters")
        ),
        "trainable_fraction": run_summary.get("parameters", {}).get(
            "trainable_fraction", spec.get("trainable_fraction")
        ),
        "final_model_size_gib": final_model_size / (1024**3),
        "checkpoint_size_gib": checkpoint_size / (1024**3),
    }


def format_value(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    headers = list(frame.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(format_value(row[col]) for col in headers) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_comparison(output_dir: str | Path) -> pd.DataFrame:
    rows = [row for spec in DEFAULT_EXPERIMENTS if (row := row_from_experiment(spec)) is not None]
    frame = pd.DataFrame(rows)
    out_dir = ensure_dir(output_dir)
    frame.to_csv(out_dir / "experiment_comparison.csv", index=False)
    write_markdown(frame, out_dir / "experiment_comparison.md")
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/visualizations/experiment_comparison")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_comparison(args.output_dir)
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
