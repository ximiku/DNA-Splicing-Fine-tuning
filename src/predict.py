"""Single-position DNABERT2 splice-site inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .model_utils import disable_remote_flash_attention
from .sequence_utils import get_centered_window, is_clean_dna, open_fasta


LABEL_NAMES = ["non_splice", "donor", "acceptor"]


def predict_position(
    model_dir: str | Path,
    fasta: str | Path,
    chrom: str,
    pos: int,
    strand: str,
    window_size: int = 401,
) -> dict:
    """Predict splice-site probabilities for one 1-based genomic position."""
    genome = open_fasta(fasta)
    window = get_centered_window(genome, chrom, pos, strand, window_size)
    if window is None:
        raise ValueError("Requested window is out of FASTA bounds or chromosome is missing")
    if not is_clean_dna(window.sequence):
        raise ValueError("Requested window contains non-ACGT bases")

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
    disable_remote_flash_attention(model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    inputs = tokenizer(window.sequence, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    return {
        "chrom": chrom,
        "pos": pos,
        "strand": strand,
        "window_start": window.start_1based,
        "window_end": window.end_1based,
        "sequence": window.sequence,
        "probabilities": {name: float(probs[idx]) for idx, name in enumerate(LABEL_NAMES)},
        "splice_site_probability": float(probs[1] + probs[2]),
        "pred_label": LABEL_NAMES[int(probs.argmax())],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--fasta", required=True)
    parser.add_argument("--chrom", required=True)
    parser.add_argument("--pos", required=True, type=int)
    parser.add_argument("--strand", required=True, choices=["+", "-"])
    parser.add_argument("--window_size", default=401, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_position(args.model_dir, args.fasta, args.chrom, args.pos, args.strand, args.window_size)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
