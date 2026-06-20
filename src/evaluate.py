"""Evaluate saved model or baseline predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from .config_utils import write_json
from .metrics_utils import LABEL_NAMES, classification_metrics


def probability_columns(frame: pd.DataFrame) -> list[str]:
    """Find probability columns in canonical class order."""
    columns = []
    for idx, name in enumerate(LABEL_NAMES):
        expected = f"prob_{idx}_{name}"
        if expected in frame.columns:
            columns.append(expected)
        elif f"prob_{idx}" in frame.columns:
            columns.append(f"prob_{idx}")
        else:
            raise ValueError(f"Missing probability column for class {idx} ({name})")
    return columns


def evaluate_predictions(predictions: Path) -> dict:
    """Compute overall and negative-type metrics from a predictions table."""
    if predictions.suffix == ".parquet":
        frame = pd.read_parquet(predictions)
    elif predictions.suffix == ".jsonl":
        frame = pd.read_json(predictions, lines=True)
    else:
        frame = pd.read_csv(predictions)
    prob_cols = probability_columns(frame)
    metrics = {"overall": classification_metrics(frame["label"].to_numpy(), frame[prob_cols].to_numpy())}

    if "negative_type" in frame.columns:
        metrics["negative_type"] = {}
        for negative_type, group in frame[frame["label"] == 0].groupby("negative_type"):
            if not negative_type:
                continue
            metrics["negative_type"][str(negative_type)] = classification_metrics(
                group["label"].to_numpy(),
                group[prob_cols].to_numpy(),
            )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Parquet/CSV/JSONL predictions table")
    parser.add_argument("--output", default=None, help="Optional metrics JSON output path")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    metrics = evaluate_predictions(Path(args.predictions))
    if args.output:
        write_json(metrics, args.output)
    else:
        import json

        print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

