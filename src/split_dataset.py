"""Standalone splitter for already materialized splice-site tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from .build_splice_dataset import split_dataframe
from .config_utils import ensure_dir, load_config, set_seed, setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True, help="Input parquet/jsonl/csv table with sequence and label columns")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    set_seed(int(config.get("seed", 42)))

    input_path = Path(args.input)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".jsonl":
        df = pd.read_json(input_path, lines=True)
    else:
        df = pd.read_csv(input_path)

    frames = split_dataframe(df, config.get("split", {}), int(config.get("seed", 42)))
    out = ensure_dir(args.output_dir)
    dataset = DatasetDict({name: Dataset.from_pandas(frame, preserve_index=False) for name, frame in frames.items()})
    dataset.save_to_disk(str(out))
    for name, frame in frames.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    write_json({"split_sizes": {name: len(frame) for name, frame in frames.items()}}, out / "split_summary.json")


if __name__ == "__main__":
    main()

