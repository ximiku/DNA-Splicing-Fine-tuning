"""Batch DNABERT2 predictions for a saved sequence table."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config_utils import ensure_dir, write_json
from .metrics_utils import LABEL_NAMES, classification_metrics
from .model_utils import disable_remote_flash_attention


def read_table(path: Path) -> pd.DataFrame:
    """Read a sequence table from parquet, JSONL, or CSV."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def predict_frame(
    frame: pd.DataFrame,
    model_dir: str | Path,
    batch_size: int,
    max_length: int,
    disable_flash_attention: bool = True,
) -> pd.DataFrame:
    """Append DNABERT2 class probabilities and predicted labels to a dataframe."""
    if "sequence" not in frame.columns:
        raise ValueError("Input table must contain a sequence column")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
    if disable_flash_attention:
        disable_remote_flash_attention(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    probabilities: list[np.ndarray] = []
    sequences = frame["sequence"].astype(str).tolist()
    for start in tqdm(range(0, len(sequences), batch_size), desc="Predicting"):
        batch = sequences[start : start + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        probabilities.append(probs)

    probs_all = np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, len(LABEL_NAMES)))
    out = frame.copy()
    for idx, name in enumerate(LABEL_NAMES):
        out[f"prob_{idx}_{name}"] = probs_all[:, idx]
    out["pred_label"] = np.argmax(probs_all, axis=1)
    out["splice_site_probability"] = probs_all[:, 1] + probs_all[:, 2]
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True, help="Path to a saved DNABERT2 model directory")
    parser.add_argument("--input", required=True, help="Input parquet/jsonl/csv with sequence and optional label columns")
    parser.add_argument("--output", required=True, help="Output parquet predictions path")
    parser.add_argument("--metrics_output", default=None, help="Optional metrics JSON path when labels are present")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--keep_flash_attention", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    frame = read_table(input_path)
    predictions = predict_frame(
        frame,
        args.model_dir,
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        disable_flash_attention=not bool(args.keep_flash_attention),
    )
    ensure_dir(output_path.parent)
    predictions.to_parquet(output_path, index=False)
    logging.info("Saved predictions to %s", output_path)

    if args.metrics_output and "label" in predictions.columns:
        prob_cols = [f"prob_{idx}_{name}" for idx, name in enumerate(LABEL_NAMES)]
        metrics = classification_metrics(predictions["label"].to_numpy(), predictions[prob_cols].to_numpy())
        write_json(metrics, Path(args.metrics_output))
        logging.info("Saved metrics to %s", args.metrics_output)


if __name__ == "__main__":
    main()
