"""GENCODE GTF parsing and transcript-level splice-site derivation."""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from .sequence_utils import normalize_chrom


ATTRIBUTE_RE = re.compile(r'(\S+)\s+"([^"]*)"')


@dataclass(frozen=True)
class Exon:
    """A GTF exon record with 1-based inclusive coordinates."""

    chrom: str
    start: int
    end: int
    strand: str
    gene_id: str
    transcript_id: str
    exon_number: int | None = None


@dataclass(frozen=True)
class Intron:
    """A transcript-level intron with 1-based inclusive genomic coordinates."""

    chrom: str
    start: int
    end: int
    strand: str
    gene_id: str
    transcript_id: str
    intron_index: int


def parse_gtf_attributes(attribute_text: str) -> dict[str, str]:
    """Parse the semicolon-delimited GTF attribute column."""
    return {key: value for key, value in ATTRIBUTE_RE.findall(attribute_text)}


def parse_exons_by_transcript(
    gtf_path: str | Path,
    fasta_keys: Iterable[str],
    max_transcripts: int | None = None,
    max_lines: int | None = None,
) -> dict[str, list[Exon]]:
    """Stream exon records from a GTF and group them by transcript_id."""
    fasta_key_set = set(fasta_keys)
    transcripts: dict[str, list[Exon]] = defaultdict(list)
    selected_transcripts: set[str] = set()
    skipped_chrom = 0
    parsed_exons = 0

    with Path(gtf_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(tqdm(handle, desc="Parsing GTF exons"), start=1):
            if max_lines is not None and line_number > max_lines:
                break
            if not line or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "exon":
                continue

            attributes = parse_gtf_attributes(fields[8])
            transcript_id = attributes.get("transcript_id")
            gene_id = attributes.get("gene_id")
            if not transcript_id or not gene_id:
                continue

            if max_transcripts is not None and transcript_id not in selected_transcripts:
                if len(selected_transcripts) >= max_transcripts:
                    break
                selected_transcripts.add(transcript_id)

            chrom = normalize_chrom(fields[0], fasta_key_set)
            if chrom is None:
                skipped_chrom += 1
                continue

            exon_number = attributes.get("exon_number")
            transcripts[transcript_id].append(
                Exon(
                    chrom=chrom,
                    start=int(fields[3]),
                    end=int(fields[4]),
                    strand=fields[6],
                    gene_id=gene_id,
                    transcript_id=transcript_id,
                    exon_number=int(exon_number) if exon_number and exon_number.isdigit() else None,
                )
            )
            parsed_exons += 1

    logging.info(
        "Parsed %s exon records across %s transcripts; skipped %s exons with unmatched chromosomes",
        parsed_exons,
        len(transcripts),
        skipped_chrom,
    )
    return dict(transcripts)


def transcript_introns(exons: list[Exon]) -> list[Intron]:
    """Infer introns from adjacent exons in transcript order."""
    if len(exons) < 2:
        return []
    strands = {exon.strand for exon in exons}
    chroms = {exon.chrom for exon in exons}
    if len(strands) != 1 or len(chroms) != 1:
        return []

    strand = exons[0].strand
    ordered = sorted(exons, key=lambda exon: (exon.start, exon.end), reverse=strand == "-")
    introns: list[Intron] = []
    for idx, (left_tx_exon, right_tx_exon) in enumerate(zip(ordered, ordered[1:]), start=1):
        genomic_start = min(left_tx_exon.end, right_tx_exon.end) + 1
        genomic_end = max(left_tx_exon.start, right_tx_exon.start) - 1
        if genomic_start > genomic_end:
            continue
        introns.append(
            Intron(
                chrom=left_tx_exon.chrom,
                start=genomic_start,
                end=genomic_end,
                strand=strand,
                gene_id=left_tx_exon.gene_id,
                transcript_id=left_tx_exon.transcript_id,
                intron_index=idx,
            )
        )
    return introns


def derive_introns(transcripts: dict[str, list[Exon]]) -> list[Intron]:
    """Infer introns for every transcript in a parsed GTF exon dictionary."""
    introns: list[Intron] = []
    for exons in tqdm(transcripts.values(), desc="Deriving transcript introns"):
        introns.extend(transcript_introns(exons))
    logging.info("Derived %s transcript-level introns", len(introns))
    return introns


def positive_site_records(introns: Iterable[Intron]) -> list[dict[str, object]]:
    """Create donor and acceptor records using intron-side splice-site anchors."""
    records: list[dict[str, object]] = []
    for intron in introns:
        if intron.strand == "+":
            donor_pos = intron.start
            acceptor_pos = intron.end
        else:
            donor_pos = intron.end
            acceptor_pos = intron.start

        shared = {
            "chrom": intron.chrom,
            "strand": intron.strand,
            "gene_id": intron.gene_id,
            "transcript_id": intron.transcript_id,
            "intron_start": intron.start,
            "intron_end": intron.end,
            "intron_index": intron.intron_index,
            "negative_type": "",
        }
        records.append({**shared, "pos": donor_pos, "label": 1, "label_name": "donor", "site_type": "donor"})
        records.append(
            {**shared, "pos": acceptor_pos, "label": 2, "label_name": "acceptor", "site_type": "acceptor"}
        )
    return records


def deduplicate_positive_records(records: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Deduplicate positives by chrom, pos, strand, and site type while retaining provenance."""
    grouped: dict[tuple[object, object, object, object], dict[str, object]] = {}
    transcript_ids: dict[tuple[object, object, object, object], set[str]] = defaultdict(set)
    gene_ids: dict[tuple[object, object, object, object], set[str]] = defaultdict(set)

    for record in records:
        key = (record["chrom"], record["pos"], record["strand"], record["site_type"])
        grouped.setdefault(key, dict(record))
        transcript_ids[key].add(str(record.get("transcript_id", "")))
        gene_ids[key].add(str(record.get("gene_id", "")))

    deduped: list[dict[str, object]] = []
    for key, record in grouped.items():
        record["transcript_id"] = ";".join(sorted(transcript_ids[key]))
        record["gene_id"] = ";".join(sorted(gene_ids[key]))
        record["n_transcripts"] = len(transcript_ids[key])
        record["n_genes"] = len(gene_ids[key])
        deduped.append(record)
    logging.info("Deduplicated positive sites from input records to %s records", len(deduped))
    return deduped

