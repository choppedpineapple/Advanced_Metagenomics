#!/usr/bin/env python3
"""
Iteratively extend cluster consensuses by recruiting additional reads.

For each cluster consensus:
  1. Extract left/right seed kmers.
  2. Recruit reads from the merged FASTQ that contain either seed.
  3. Filter redundant reads, append to existing cluster FASTQ.
  4. Generate a new consensus with SPOA (via conda run).

This script is tailored for the mock workflow; parameters can be tuned as needed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple


def read_fasta(path: Path) -> Dict[str, str]:
    seqs: Dict[str, str] = {}
    name = None
    buf: List[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    seqs[name] = "".join(buf)
                name = line[1:]
                buf = []
            else:
                buf.append(line)
        if name:
            seqs[name] = "".join(buf)
    return seqs


def read_fastq(path: Path) -> Dict[str, Tuple[str, str, str]]:
    records: Dict[str, Tuple[str, str, str]] = {}
    with path.open() as fh:
        while True:
            header = fh.readline().rstrip()
            if not header:
                break
            seq = fh.readline().rstrip()
            plus = fh.readline().rstrip()
            qual = fh.readline().rstrip()
            if not qual:
                raise ValueError("Malformed FASTQ; missing quality line.")
            rid = header[1:] if header.startswith("@") else header
            records[rid] = (seq, plus, qual)
    return records


def write_fastq(path: Path, records: Dict[str, Tuple[str, str, str]]) -> None:
    with path.open("w") as out:
        for rid, (seq, plus, qual) in records.items():
            out.write(f"@{rid}\n{seq}\n{plus}\n{qual}\n")


def recruit_reads(
    merged_fastq: Path,
    seeds: List[str],
    min_match: int,
) -> Dict[str, Tuple[str, str, str]]:
    recruited: Dict[str, Tuple[str, str, str]] = {}
    with merged_fastq.open() as fh:
        while True:
            header = fh.readline()
            if not header:
                break
            seq = fh.readline().strip()
            plus = fh.readline().strip()
            qual = fh.readline().strip()
            rid = header[1:].strip() if header.startswith("@") else header.strip()
            for seed in seeds:
                if seed in seq:
                    if len(seed) >= min_match:
                        recruited[rid] = (seq, plus, qual)
                        break
    return recruited


def run_spoa(input_fastq: Path, output_fasta: Path, conda_env: str) -> None:
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "spoa",
        "-r",
        "0",
        str(input_fastq),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"SPOA failed: {result.stderr}")
    with output_fasta.open("w") as out:
        out.write(result.stdout)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extend cluster consensuses via seed read recruitment.")
    parser.add_argument("--consensus-dir", required=True, type=Path, help="Directory with per-cluster consensus FASTA.")
    parser.add_argument("--cluster-fastq-dir", required=True, type=Path, help="Directory with per-cluster FASTQs.")
    parser.add_argument("--merged-fastq", required=True, type=Path, help="Merged reads FASTQ.")
    parser.add_argument("--conda-env", default="scfv", help="Conda environment for running SPOA.")
    parser.add_argument("--prefix-length", type=int, default=40, help="Prefix seed length.")
    parser.add_argument("--suffix-length", type=int, default=40, help="Suffix seed length.")
    parser.add_argument("--min-match", type=int, default=35, help="Minimum seed length match to retain read.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for updated FASTQs and consensuses.")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Combine all consensus FASTA files if a single file isn't provided.
    consensus_path = args.consensus_dir
    if consensus_path.is_file():
        consensus_sequences = read_fasta(consensus_path)
    else:
        # Concatenate all *_consensus.fa files in the directory.
        combined = {}
        for fasta in sorted(consensus_path.glob("*_consensus.fa")):
            combined.update(read_fasta(fasta))
        if not combined:
            raise FileNotFoundError("No consensus FASTA files found in the provided directory.")
        consensus_sequences = combined

    for cluster_name, seq in consensus_sequences.items():
        prefix = seq[: args.prefix_length]
        suffix = seq[-args.suffix_length :]
        seeds = [prefix, suffix]
        print(f"{cluster_name}: seeds {prefix[:10]}..., ...{suffix[-10:]}")

        recruited = recruit_reads(args.merged_fastq, seeds, args.min_match)
        print(f"{cluster_name}: recruited {len(recruited)} reads from merged dataset.")

        original_fastq_path = args.cluster_fastq_dir / f"{cluster_name}.fastq"
        original_records = read_fastq(original_fastq_path)
        merged_records = dict(original_records)
        merged_records.update(recruited)  # maintain original reads, add new ones

        out_fastq = args.output_dir / f"{cluster_name}.fastq"
        write_fastq(out_fastq, merged_records)
        print(f"{cluster_name}: wrote {len(merged_records)} reads to {out_fastq}")

        out_consensus = args.output_dir / f"{cluster_name}_consensus.fa"
        run_spoa(out_fastq, out_consensus, args.conda_env)
        # Normalize header to cluster name
        consensus_seq = "".join(
            line.strip() for line in out_consensus.open() if not line.startswith(">")
        )
        with out_consensus.open("w") as handle:
            handle.write(f">{cluster_name}\n{consensus_seq}\n")
        print(f"{cluster_name}: updated consensus written to {out_consensus}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
