#!/usr/bin/env python3
"""
Extract linker-containing reads, orient them, and emit VH/VL segments.

This script scans a merged FASTQ file for a specified linker sequence,
optionally allowing mismatches. Reads are re-oriented so that the linker
is in the forward orientation and flanking segments are written out for
subsequent clustering and assembly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Tuple

BASES = "ACGTN"
RC_TABLE = str.maketrans("ACGTNacgtn", "TGCANtgcan")


@dataclass
class FastqRecord:
    header: str
    sequence: str
    plus: str
    quality: str


def parse_fastq(handle) -> Iterator[FastqRecord]:
    while True:
        header = handle.readline().rstrip()
        if not header:
            return
        sequence = handle.readline().rstrip()
        plus = handle.readline().rstrip()
        quality = handle.readline().rstrip()
        if not quality:
            raise ValueError("Malformed FASTQ; missing quality line.")
        yield FastqRecord(header, sequence, plus, quality)


def reverse_complement(seq: str) -> str:
    return seq.translate(RC_TABLE)[::-1]


def reverse_quality(qual: str) -> str:
    return qual[::-1]


def hamming_distance(a: str, b: str) -> int:
    mismatches = 0
    for x, y in zip(a, b):
        if y == "N":
            continue
        if x == "N":
            mismatches += 1
            continue
        if x != y:
            mismatches += 1
    return mismatches


def find_linker(seq: str, linker: str, max_mismatches: int) -> Tuple[int, int] | None:
    """Return (position, mismatches) for the first acceptable match."""
    window = len(linker)
    limit = len(seq) - window + 1
    for idx in range(limit):
        candidate = seq[idx : idx + window]
        mismatches = hamming_distance(candidate, linker)
        if mismatches <= max_mismatches:
            return idx, mismatches
    return None


def canon_oriented_record(
    record: FastqRecord, linker: str, max_mismatches: int
) -> Tuple[FastqRecord, int, int] | None:
    """
    Orient record so linker appears in forward orientation.

    Returns (record, position, mismatches) in canonical orientation.
    """
    fwd = find_linker(record.sequence, linker, max_mismatches)
    if fwd:
        pos, mism = fwd
        return record, pos, mism

    rc_seq = reverse_complement(record.sequence)
    rc_qual = reverse_quality(record.quality)
    rc_rec = FastqRecord(record.header, rc_seq, record.plus, rc_qual)
    rev = find_linker(rc_seq, linker, max_mismatches)
    if rev:
        pos, mism = rev
        return rc_rec, pos, mism
    return None


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def write_record(handle, record: FastqRecord) -> None:
    handle.write(f"{record.header}\n{record.sequence}\n{record.plus}\n{record.quality}\n")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extract linker-containing reads and their flanking VH/VL sequences."
    )
    parser.add_argument("fastq", type=Path, help="Merged FASTQ file.")
    parser.add_argument("output", type=Path, help="Output directory.")
    parser.add_argument("--linker", required=True, help="Linker sequence (forward orientation).")
    parser.add_argument("--max-mismatches", type=int, default=2, help="Maximum linker mismatches.")
    parser.add_argument("--vh-window", type=int, default=160, help="Upstream window to retain.")
    parser.add_argument("--vh-min", type=int, default=60, help="Minimum upstream bases required.")
    parser.add_argument("--cdr3-offset", type=int, default=60, help="Distance upstream of linker for CDR3 window.")
    parser.add_argument("--cdr3-length", type=int, default=60, help="Length of CDR3 window to extract.")
    parser.add_argument("--vl-window", type=int, default=200, help="Downstream window length to retain.")
    args = parser.parse_args(argv)

    if args.cdr3_offset < args.cdr3_length:
        parser.error("--cdr3-offset must be >= --cdr3-length to keep CDR3 fully upstream of linker.")

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    hits_tsv = (out_dir / "linker_hits.tsv").open("w")
    oriented_fastq = (out_dir / "linker_oriented.fastq").open("w")
    vh_fa = (out_dir / "vh_upstream.fa").open("w")
    cdr3_fa = (out_dir / "vh_cdr3.fa").open("w")
    vl_fa = (out_dir / "vl_downstream.fa").open("w")

    hits_tsv.write("read_id\tposition\tmismatches\n")

    total = matched = 0
    with args.fastq.open() as handle:
        for record in parse_fastq(handle):
            total += 1
            canon = canon_oriented_record(record, args.linker, args.max_mismatches)
            if not canon:
                continue
            matched += 1
            canon_record, pos, mism = canon
            read_id = canon_record.header[1:] if canon_record.header.startswith("@") else canon_record.header
            hits_tsv.write(f"{read_id}\t{pos}\t{mism}\n")
            write_record(oriented_fastq, canon_record)

            linker_start = pos
            if linker_start >= args.vh_min:
                upstream_start = clamp(linker_start - args.vh_window, 0, linker_start)
                upstream_seq = canon_record.sequence[upstream_start:linker_start]
                if len(upstream_seq) >= args.vh_min:
                    vh_fa.write(f">{read_id}\n{upstream_seq}\n")

                cdr3_start = linker_start - args.cdr3_offset
                cdr3_end = cdr3_start + args.cdr3_length
                if cdr3_start >= 0 and cdr3_end <= len(canon_record.sequence):
                    cdr3_seq = canon_record.sequence[cdr3_start:cdr3_end]
                    if len(cdr3_seq) == args.cdr3_length:
                        cdr3_fa.write(f">{read_id}\n{cdr3_seq}\n")

            vl_start = linker_start + len(args.linker)
            vl_end = clamp(vl_start + args.vl_window, vl_start, len(canon_record.sequence))
            vl_seq = canon_record.sequence[vl_start:vl_end]
            if vl_seq:
                vl_fa.write(f">{read_id}\n{vl_seq}\n")

    hits_tsv.close()
    oriented_fastq.close()
    vh_fa.close()
    cdr3_fa.close()
    vl_fa.close()

    print(f"Processed {total} reads; matched {matched} ({matched/total:.2%}).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
