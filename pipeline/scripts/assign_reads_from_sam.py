#!/usr/bin/env python3
"""
Assign reads to clusters based on SAM alignments and emit per-cluster FASTQ files.

The SAM file is expected to contain single best alignments (e.g. BBMap with ambig=best).
Only mapped reads are considered. Unmapped reads are ignored.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple


def parse_sam(path: Path) -> Iterator[Tuple[str, str, int]]:
    """Yield (read_id, reference_name, edit_distance)."""
    with path.open() as handle:
        for line in handle:
            if not line or line.startswith("@"):
                continue
            fields = line.rstrip().split("\t")
            if len(fields) < 11:
                continue
            qname = fields[0]
            flag = int(fields[1])
            rname = fields[2]
            if flag & 0x4 or rname == "*":
                continue
            nm = None
            for field in fields[11:]:
                if field.startswith("NM:i:"):
                    nm = int(field.split(":")[2])
                    break
            if nm is None:
                nm = 0
            yield qname, rname, nm


def load_fastq(path: Path) -> Dict[str, Tuple[str, str, str]]:
    records: Dict[str, Tuple[str, str, str]] = {}
    with path.open() as handle:
        while True:
            header = handle.readline().rstrip()
            if not header:
                break
            seq = handle.readline().rstrip()
            plus = handle.readline().rstrip()
            qual = handle.readline().rstrip()
            if not qual:
                raise ValueError("Malformed FASTQ; missing quality line.")
            rid = header[1:] if header.startswith("@") else header
            records[rid] = (seq, plus, qual)
    return records


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Assign reads to clusters based on SAM alignments.")
    parser.add_argument("--sam", required=True, type=Path, help="SAM file with alignments.")
    parser.add_argument("--fastq", required=True, type=Path, help="Original FASTQ containing the reads.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for per-cluster FASTQ files.")
    parser.add_argument("--max-nm", type=int, default=8, help="Maximum edit distance (NM) to keep alignment.")
    args = parser.parse_args(argv)

    assignments: Dict[str, Tuple[str, int]] = {}
    for read_id, ref_name, nm in parse_sam(args.sam):
        if nm > args.max_nm:
            continue
        # Keep the best (lowest NM) per read
        if read_id not in assignments or nm < assignments[read_id][1]:
            assignments[read_id] = (ref_name, nm)

    if not assignments:
        print("No reads met the alignment criteria.")
        return 0

    fastq_records = load_fastq(args.fastq)
    by_cluster: Dict[str, Dict[str, Tuple[str, str, str]]] = defaultdict(dict)
    for read_id, (ref_name, nm) in assignments.items():
        if read_id in fastq_records:
            by_cluster[ref_name][read_id] = fastq_records[read_id]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for ref_name, recs in by_cluster.items():
        out_path = args.output_dir / f"{ref_name}.fastq"
        with out_path.open("w") as out:
            for rid, (seq, plus, qual) in sorted(recs.items()):
                out.write(f"@{rid}\n{seq}\n{plus}\n{qual}\n")
        print(f"{ref_name}: wrote {len(recs)} reads to {out_path}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
