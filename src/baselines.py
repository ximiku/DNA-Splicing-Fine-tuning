"""Classical baseline models for splice-site prediction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config_utils import ensure_dir, load_config, set_seed, setup_logging, write_json
from .metrics_utils import LABEL_NAMES, classification_metrics


def aligned_predict_proba(clf: Pipeline, sequences: pd.Series) -> pd.DataFrame:
    """Return predict_proba output aligned to labels 0, 1, and 2."""
    raw_probs = clf.predict_proba(sequences)
    classes = list(clf.named_steps["model"].classes_)
    aligned = pd.DataFrame(0.0, index=range(len(sequences)), columns=[0, 1, 2])
    for raw_idx, class_label in enumerate(classes):
        aligned[int(class_label)] = raw_probs[:, raw_idx]
    return aligned


def run_baseline(config: dict) -> dict:
    """Train a char n-gram logistic regression baseline and save metrics."""
    set_seed(int(config.get("seed", 42)))
    output_dir = ensure_dir(config["output"]["output_dir"])
    dataset = load_from_disk(config["data"]["dataset_dir"])

    if config.get("smoke", {}).get("max_examples_per_split"):
        limit = int(config["smoke"]["max_examples_per_split"])
        dataset = {split: ds.shuffle(seed=42).select(range(min(limit, len(ds)))) for split, ds in dataset.items()}

    ngram_min, ngram_max = config.get("baseline", {}).get("ngram_range", [3, 6])
    max_features = config.get("baseline", {}).get("max_features", 200000)
    clf = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char",
                    ngram_range=(int(ngram_min), int(ngram_max)),
                    lowercase=False,
                    max_features=max_features,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=int(config.get("baseline", {}).get("max_iter", 1000)),
                    class_weight=config.get("baseline", {}).get("class_weight", "balanced"),
                ),
            ),
        ]
    )

    train = dataset["train"].to_pandas()
    valid = dataset["valid"].to_pandas()
    test = dataset["test"].to_pandas()
    logging.info("Training baseline: train=%s valid=%s test=%s", len(train), len(valid), len(test))
    clf.fit(train["sequence"], train["label"])

    metrics = {}
    predictions = {}
    for split_name, frame in {"valid": valid, "test": test}.items():
        probs_frame = aligned_predict_proba(clf, frame["sequence"])
        probs = probs_frame[[0, 1, 2]].to_numpy()
        metrics[split_name] = classification_metrics(frame["label"].to_numpy(), probs)
        pred_frame = frame.copy()
        for idx, name in enumerate(LABEL_NAMES):
            pred_frame[f"prob_{idx}_{name}"] = probs[:, idx]
        pred_frame["pred_label"] = probs.argmax(axis=1)
        pred_frame["splice_site_probability"] = probs[:, 1] + probs[:, 2]
        predictions[split_name] = pred_frame
        pred_frame.to_parquet(output_dir / f"{split_name}_predictions.parquet", index=False)

    joblib.dump(clf, output_dir / "tfidf_logreg.joblib")
    write_json(metrics, output_dir / "metrics.json")
    logging.info("Saved baseline artifacts to %s", output_dir)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    run_baseline(config)


if __name__ == "__main__":
    main()
