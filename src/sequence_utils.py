"""FASTA access, coordinate conversion, and strand orientation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pyfaidx import Fasta


DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass(frozen=True)
class Window:
    """A centered genomic window using 1-based inclusive genomic coordinates."""

    chrom: str
    center_pos: int
    strand: str
    start_1based: int
    end_1based: int
    sequence: str


def reverse_complement(sequence: str) -> str:
    """Return the reverse complement of a DNA sequence."""
    return sequence.translate(DNA_COMPLEMENT)[::-1].upper()


def open_fasta(path: str | Path) -> Fasta:
    """Open an indexed FASTA file with pyfaidx."""
    return Fasta(str(path), as_raw=False, sequence_always_upper=True)


def normalize_chrom(chrom: str, fasta_keys: set[str]) -> str | None:
    """Match GTF chromosome names to FASTA names, handling optional chr prefix."""
    if chrom in fasta_keys:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in fasta_keys:
        return chrom[3:]
    prefixed = f"chr{chrom}"
    if prefixed in fasta_keys:
        return prefixed
    return None


def chrom_lengths(genome: Fasta) -> dict[str, int]:
    """Return chromosome lengths from an opened pyfaidx FASTA."""
    return {chrom: len(genome[chrom]) for chrom in genome.keys()}


def get_centered_window(
    genome: Fasta,
    chrom: str,
    pos_1based: int,
    strand: str,
    window_size: int = 401,
) -> Window | None:
    """Extract a centered DNA window and orient it to transcript direction.

    Coordinates from GTF are 1-based inclusive. pyfaidx slicing is Python-style
    0-based half-open, so genomic [start_1based, end_1based] becomes
    genome[chrom][start_1based - 1:end_1based].
    """
    if window_size % 2 != 1:
        raise ValueError("window_size must be odd so the candidate site is centered")
    if strand not in {"+", "-"}:
        raise ValueError("strand must be '+' or '-'")
    if chrom not in genome.keys():
        return None

    flank = window_size // 2
    start_1based = pos_1based - flank
    end_1based = pos_1based + flank
    if start_1based < 1 or end_1based > len(genome[chrom]):
        return None

    sequence = genome[chrom][start_1based - 1 : end_1based].seq.upper()
    if len(sequence) != window_size:
        return None
    if strand == "-":
        sequence = reverse_complement(sequence)
    return Window(
        chrom=chrom,
        center_pos=pos_1based,
        strand=strand,
        start_1based=start_1based,
        end_1based=end_1based,
        sequence=sequence,
    )


def is_clean_dna(sequence: str) -> bool:
    """Return True when a sequence contains only A/C/G/T."""
    return set(sequence.upper()) <= {"A", "C", "G", "T"}


def has_splice_like_motif(sequence: str, center_index: int | None = None) -> bool:
    """Check whether an oriented window has donor-like GT or acceptor-like AG near center."""
    sequence = sequence.upper()
    if center_index is None:
        center_index = len(sequence) // 2
    donor_like = sequence[center_index : center_index + 2] == "GT"
    acceptor_like = sequence[center_index - 1 : center_index + 1] == "AG"
    return donor_like or acceptor_like

