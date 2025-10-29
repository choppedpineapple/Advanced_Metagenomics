#!/usr/bin/env python3
"""
Group oriented reads by CDR3 cluster and emit per-cluster FASTQ sets.

This consumes a VSEARCH .uc assignment file, the oriented FASTQ produced by
`extract_linker_regions.py`, and writes cluster FASTQ files for downstream
assembly. Clusters below a minimum support threshold are skipped.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Set, Tuple


def parse_uc(path: Path) -> Iterator[Tuple[str, str]]:
    """Yield (query_id, cluster_id) pairs for hits."""
    with path.open() as handle:
        for line in handle:
            if not line or line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 10:
                continue
            record_type = parts[0]
            if record_type not in {"H", "S"}:
                continue
            query = parts[8]
            cluster = parts[9]
            yield query, cluster


def load_cluster_sizes(centroids: Path) -> List[Tuple[str, int]]:
    records: List[Tuple[str, int]] = []
    with centroids.open() as handle:
        for line in handle:
            if line.startswith(">"):
                header = line[1:].strip()
                fields = header.split(";")
                name = fields[0]
                size_field = next((f for f in fields[1:] if f.startswith("size=")), None)
                size = int(size_field.split("=")[1]) if size_field else 1
                records.append((name, size))
    return records


def collect_read_sequences(fastq_path: Path, keep: Set[str]) -> Dict[str, Tuple[str, str, str]]:
    seqs: Dict[str, Tuple[str, str, str]] = {}
    with fastq_path.open() as handle:
        while True:
            header = handle.readline().rstrip()
            if not header:
                break
            seq = handle.readline().rstrip()
            plus = handle.readline().rstrip()
            qual = handle.readline().rstrip()
            if not qual:
                raise ValueError("Malformed FASTQ; missing quality line.")
            read_id = header[1:] if header.startswith("@") else header
            if read_id in keep:
                seqs[read_id] = (seq, plus, qual)
    return seqs


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Partition oriented reads by cluster.")
    parser.add_argument("--uc", required=True, type=Path, help="VSEARCH .uc assignments.")
    parser.add_argument(
        "--centroids",
        required=True,
        type=Path,
        help="Centroid FASTA produced by VSEARCH with size annotations.",
    )
    parser.add_argument(
        "--oriented-fastq",
        required=True,
        type=Path,
        help="FASTQ from extract_linker_regions.py (canonical orientation).",
    )
    parser.add_argument("--output", required=True, type=Path, help="Cluster FASTQ output directory.")
    parser.add_argument("--min-size", type=int, default=20, help="Minimum cluster size to retain.")
    parser.add_argument("--label-prefix", default="vh_cluster_", help="Prefix for cluster labels.")
    args = parser.parse_args(argv)

    cluster_records = load_cluster_sizes(args.centroids)
    label_mapping: Dict[str, Tuple[str, int]] = {}
    for idx, (orig_name, size) in enumerate(cluster_records, start=1):
        if size >= args.min_size:
            label = f"{args.label_prefix}{idx}"
            label_mapping[orig_name] = (label, size)

    assignments: Dict[str, Set[str]] = defaultdict(set)
    for query, cluster in parse_uc(args.uc):
        if cluster in label_mapping:
            assignments[cluster].add(query)

    if not assignments:
        print("No clusters met the minimum size requirement.", file=sys.stderr)
        return 0

    all_read_ids = set().union(*assignments.values())
    read_cache = collect_read_sequences(args.oriented_fastq, all_read_ids)

    args.output.mkdir(parents=True, exist_ok=True)
    manifest_lines = ["cluster_id\tsize\tcentroid_read"]

    for orig_name, (label, size) in label_mapping.items():
        if orig_name not in assignments:
            continue
        reads = assignments[orig_name]
        out_path = args.output / f"{label}.fastq"
        with out_path.open("w") as out:
            for rid in sorted(reads):
                if rid not in read_cache:
                    continue
                seq, plus, qual = read_cache[rid]
                out.write(f"@{rid}\n{seq}\n{plus}\n{qual}\n")
        manifest_lines.append(f"{label}\t{size}\t{orig_name}")
        print(f"{label}: wrote {len(reads)} reads to {out_path} (centroid {orig_name}, size {size})")

    manifest_path = args.output / "cluster_manifest.tsv"
    with manifest_path.open("w") as manifest:
        manifest.write("\n".join(manifest_lines) + "\n")
    print(f"Wrote manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())
