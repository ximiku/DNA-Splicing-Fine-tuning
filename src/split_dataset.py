"""Standalone splitter for already materialized splice-site tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict

from .build_splice_dataset import split_dataframe, split_summary
from .config_utils import ensure_dir, load_config, set_seed, setup_logging, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--input",
        required=True,
        nargs="+",
        help="One or more parquet/jsonl/csv tables with sequence and label columns",
    )
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def read_table(path: Path) -> pd.DataFrame:
    """Read a dataframe from parquet, JSONL, or CSV."""
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    set_seed(int(config.get("seed", 42)))

    frames_in = [read_table(Path(path)) for path in args.input]
    df = pd.concat(frames_in, ignore_index=True)

    frames = split_dataframe(df, config.get("split", {}), int(config.get("seed", 42)))
    out = ensure_dir(args.output_dir)
    dataset = DatasetDict({name: Dataset.from_pandas(frame, preserve_index=False) for name, frame in frames.items()})
    dataset.save_to_disk(str(out))
    for name, frame in frames.items():
        frame.to_parquet(out / f"{name}.parquet", index=False)
    write_json({"split": config.get("split", {}), **split_summary(frames)}, out / "split_summary.json")


if __name__ == "__main__":
    main()
