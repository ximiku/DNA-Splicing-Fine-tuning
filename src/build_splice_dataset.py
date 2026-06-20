"""Build a DNABERT2-ready splice-site dataset from FASTA and GENCODE GTF."""

from __future__ import annotations

import argparse
import bisect
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config_utils import ensure_dir, load_config, set_seed, setup_logging, write_json
from .gtf_utils import (
    Intron,
    deduplicate_positive_records,
    derive_introns,
    parse_exons_by_transcript,
    positive_site_records,
)
from .sequence_utils import chrom_lengths, get_centered_window, has_splice_like_motif, is_clean_dna, open_fasta


LABEL_NAMES = {0: "non_splice", 1: "donor", 2: "acceptor"}


def positive_records_with_sequences(
    records: list[dict[str, Any]],
    genome: Any,
    window_size: int,
) -> tuple[list[dict[str, Any]], Counter]:
    """Attach oriented sequence windows to positive site records."""
    kept: list[dict[str, Any]] = []
    counters: Counter = Counter()
    for record in tqdm(records, desc="Extracting positive windows"):
        window = get_centered_window(genome, str(record["chrom"]), int(record["pos"]), str(record["strand"]), window_size)
        if window is None:
            counters["window_out_of_bounds"] += 1
            continue
        if not is_clean_dna(window.sequence):
            counters["contains_N"] += 1
            continue
        kept.append(
            {
                **record,
                "sequence": window.sequence,
                "window_start": window.start_1based,
                "window_end": window.end_1based,
            }
        )
        counters["kept"] += 1
    return kept, counters


def build_known_site_index(records: list[dict[str, Any]]) -> dict[str, list[int]]:
    """Build a sorted chromosome -> positions map for distance checks."""
    index: dict[str, set[int]] = defaultdict(set)
    for record in records:
        index[str(record["chrom"])].add(int(record["pos"]))
    return {chrom: sorted(positions) for chrom, positions in index.items()}


def near_known_site(chrom: str, pos: int, known_index: dict[str, list[int]], min_distance: int) -> bool:
    """Return True when pos is within min_distance of an annotated splice site."""
    positions = known_index.get(chrom)
    if not positions:
        return False
    insert_at = bisect.bisect_left(positions, pos)
    if insert_at < len(positions) and abs(positions[insert_at] - pos) <= min_distance:
        return True
    if insert_at > 0 and abs(positions[insert_at - 1] - pos) <= min_distance:
        return True
    return False


def weighted_chrom_sampler(lengths: dict[str, int], min_len: int) -> tuple[list[str], list[int]]:
    """Return chromosomes and positive sampling weights by chromosome length."""
    chroms = [chrom for chrom, length in lengths.items() if length >= min_len]
    weights = [lengths[chrom] for chrom in chroms]
    return chroms, weights


def make_negative_record(
    genome: Any,
    chrom: str,
    pos: int,
    strand: str,
    window_size: int,
    negative_type: str,
    known_index: dict[str, list[int]],
    min_distance: int,
    used_keys: set[tuple[str, int, str]],
    require_motif: bool = False,
) -> dict[str, Any] | None:
    """Validate one candidate negative and return a dataset record."""
    key = (chrom, pos, strand)
    if key in used_keys:
        return None
    if near_known_site(chrom, pos, known_index, min_distance):
        return None
    window = get_centered_window(genome, chrom, pos, strand, window_size)
    if window is None or not is_clean_dna(window.sequence):
        return None
    if require_motif and not has_splice_like_motif(window.sequence):
        return None
    used_keys.add(key)
    return {
        "chrom": chrom,
        "pos": pos,
        "strand": strand,
        "label": 0,
        "label_name": "non_splice",
        "site_type": "non_splice",
        "negative_type": negative_type,
        "gene_id": "",
        "transcript_id": "",
        "intron_start": -1,
        "intron_end": -1,
        "intron_index": -1,
        "n_transcripts": 0,
        "n_genes": 0,
        "sequence": window.sequence,
        "window_start": window.start_1based,
        "window_end": window.end_1based,
    }


def sample_random_negatives(
    genome: Any,
    target_count: int,
    window_size: int,
    known_index: dict[str, list[int]],
    min_distance: int,
    used_keys: set[tuple[str, int, str]],
    require_motif: bool = False,
    negative_type: str = "random_genome",
    max_attempt_multiplier: int = 200,
) -> list[dict[str, Any]]:
    """Sample random genomic negatives, optionally requiring a splice-like motif."""
    lengths = chrom_lengths(genome)
    chroms, weights = weighted_chrom_sampler(lengths, window_size)
    flank = window_size // 2
    records: list[dict[str, Any]] = []
    max_attempts = max(target_count * max_attempt_multiplier, 1000)
    attempts = 0
    with tqdm(total=target_count, desc=f"Sampling {negative_type}") as progress:
        while len(records) < target_count and attempts < max_attempts:
            attempts += 1
            chrom = random.choices(chroms, weights=weights, k=1)[0]
            pos = random.randint(flank + 1, lengths[chrom] - flank)
            strand = random.choice(["+", "-"])
            record = make_negative_record(
                genome=genome,
                chrom=chrom,
                pos=pos,
                strand=strand,
                window_size=window_size,
                negative_type=negative_type,
                known_index=known_index,
                min_distance=min_distance,
                used_keys=used_keys,
                require_motif=require_motif,
            )
            if record:
                records.append(record)
                progress.update(1)
    if len(records) < target_count:
        logging.warning("Only sampled %s/%s %s negatives", len(records), target_count, negative_type)
    return records


def sample_intronic_negatives(
    genome: Any,
    introns: list[Intron],
    target_count: int,
    window_size: int,
    known_index: dict[str, list[int]],
    min_distance: int,
    intronic_margin: int,
    used_keys: set[tuple[str, int, str]],
    max_attempt_multiplier: int = 200,
) -> list[dict[str, Any]]:
    """Sample non-splice negatives from transcript-level intron interiors."""
    eligible = [intron for intron in introns if intron.end - intron.start + 1 > 2 * intronic_margin + 1]
    records: list[dict[str, Any]] = []
    max_attempts = max(target_count * max_attempt_multiplier, 1000)
    attempts = 0
    with tqdm(total=target_count, desc="Sampling intronic") as progress:
        while len(records) < target_count and attempts < max_attempts and eligible:
            attempts += 1
            intron = random.choice(eligible)
            pos = random.randint(intron.start + intronic_margin, intron.end - intronic_margin)
            record = make_negative_record(
                genome=genome,
                chrom=intron.chrom,
                pos=pos,
                strand=intron.strand,
                window_size=window_size,
                negative_type="intronic",
                known_index=known_index,
                min_distance=min_distance,
                used_keys=used_keys,
                require_motif=False,
            )
            if record:
                record["gene_id"] = intron.gene_id
                record["transcript_id"] = intron.transcript_id
                record["intron_start"] = intron.start
                record["intron_end"] = intron.end
                record["intron_index"] = intron.intron_index
                records.append(record)
                progress.update(1)
    if len(records) < target_count:
        logging.warning("Only sampled %s/%s intronic negatives", len(records), target_count)
    return records


def negative_quotas(total_positive: int, config: dict[str, Any]) -> dict[str, int]:
    """Calculate per-type negative quotas from config."""
    negative_cfg = config.get("negatives", {})
    total = int(round(total_positive * float(negative_cfg.get("negative_to_positive_ratio", 1.0))))
    fractions = negative_cfg.get("type_fractions", {"random_genome": 1 / 3, "motif_hard": 1 / 3, "intronic": 1 / 3})
    quotas = {name: int(total * float(frac)) for name, frac in fractions.items()}
    remainder = total - sum(quotas.values())
    for name in list(quotas)[:remainder]:
        quotas[name] += 1
    return quotas


def split_dataframe(df: pd.DataFrame, split_cfg: dict[str, Any], seed: int) -> dict[str, pd.DataFrame]:
    """Random stratified train/valid/test split with fallback for tiny smoke runs."""
    test_size = float(split_cfg.get("test_size", 0.1))
    valid_size = float(split_cfg.get("valid_size", 0.1))
    stratify_col = split_cfg.get("stratify_column", "stratify_key")

    df = df.copy()
    if stratify_col == "stratify_key" and "stratify_key" not in df.columns:
        df["stratify_key"] = df.apply(
            lambda row: row["negative_type"] if row["label"] == 0 else row["label_name"],
            axis=1,
        )

    def safe_stratify(frame: pd.DataFrame, column: str) -> pd.Series | None:
        counts = frame[column].value_counts()
        return frame[column] if len(counts) > 1 and counts.min() >= 2 else None

    try:
        train_valid, test = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=safe_stratify(df, stratify_col),
        )
    except ValueError:
        train_valid, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label"])

    valid_relative = valid_size / (1.0 - test_size)
    try:
        train, valid = train_test_split(
            train_valid,
            test_size=valid_relative,
            random_state=seed,
            stratify=safe_stratify(train_valid, stratify_col),
        )
    except ValueError:
        train, valid = train_test_split(train_valid, test_size=valid_relative, random_state=seed, stratify=None)

    return {
        "train": train.reset_index(drop=True),
        "valid": valid.reset_index(drop=True),
        "test": test.reset_index(drop=True),
    }


def build_dataset(config: dict[str, Any]) -> dict[str, Any]:
    """Build and save the processed splice-site dataset."""
    seed = int(config.get("seed", 42))
    set_seed(seed)

    fasta_path = Path(config["raw"]["fasta"])
    gtf_path = Path(config["raw"]["gtf"])
    output_dir = ensure_dir(config["output"]["dataset_dir"])
    window_size = int(config.get("window_size", 401))

    genome = open_fasta(fasta_path)
    transcripts = parse_exons_by_transcript(
        gtf_path,
        genome.keys(),
        max_transcripts=config.get("limits", {}).get("max_transcripts"),
        max_lines=config.get("limits", {}).get("max_gtf_lines"),
    )
    introns = derive_introns(transcripts)
    positive_sites = deduplicate_positive_records(positive_site_records(introns))
    positive_records, positive_filter_counts = positive_records_with_sequences(positive_sites, genome, window_size)
    known_index = build_known_site_index(positive_sites)

    quotas = negative_quotas(len(positive_records), config)
    negative_cfg = config.get("negatives", {})
    min_distance = int(negative_cfg.get("min_distance_to_splice", window_size // 2))
    intronic_margin = int(negative_cfg.get("intronic_margin", max(window_size // 2, min_distance)))
    used_negative_keys: set[tuple[str, int, str]] = set()
    negatives: list[dict[str, Any]] = []
    if quotas.get("random_genome", 0) > 0:
        negatives.extend(
            sample_random_negatives(
                genome, quotas["random_genome"], window_size, known_index, min_distance, used_negative_keys
            )
        )
    if quotas.get("motif_hard", 0) > 0:
        negatives.extend(
            sample_random_negatives(
                genome,
                quotas["motif_hard"],
                window_size,
                known_index,
                min_distance,
                used_negative_keys,
                require_motif=True,
                negative_type="motif_hard",
            )
        )
    if quotas.get("intronic", 0) > 0:
        negatives.extend(
            sample_intronic_negatives(
                genome,
                introns,
                quotas["intronic"],
                window_size,
                known_index,
                min_distance,
                intronic_margin,
                used_negative_keys,
            )
        )

    df = pd.DataFrame(positive_records + negatives)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    split_frames = split_dataframe(df, config.get("split", {}), seed)
    dataset = DatasetDict({name: Dataset.from_pandas(frame, preserve_index=False) for name, frame in split_frames.items()})
    dataset.save_to_disk(str(output_dir))

    for name, frame in split_frames.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)

    summary = {
        "dataset_dir": str(output_dir),
        "window_size": window_size,
        "n_transcripts": len(transcripts),
        "n_introns": len(introns),
        "n_positive_sites_before_window_filter": len(positive_sites),
        "positive_window_filter_counts": dict(positive_filter_counts),
        "negative_quotas": quotas,
        "n_records_total": int(len(df)),
        "label_counts": {LABEL_NAMES[int(k)]: int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "negative_type_counts": {str(k): int(v) for k, v in df["negative_type"].value_counts().items() if k},
        "split_sizes": {name: int(len(frame)) for name, frame in split_frames.items()},
        "split_label_counts": {
            name: {LABEL_NAMES[int(k)]: int(v) for k, v in frame["label"].value_counts().sort_index().items()}
            for name, frame in split_frames.items()
        },
    }
    write_json(summary, output_dir / "summary.json")
    logging.info("Saved dataset to %s", output_dir)
    logging.info("Label counts: %s", summary["label_counts"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_logging(config.get("logging", {}).get("level", "INFO"))
    summary = build_dataset(config)
    logging.info("Done: %s", summary)


if __name__ == "__main__":
    main()

