#!/usr/bin/env python3
"""
End-to-end orchestration script for the scFv reconstruction workflow.

This script stitches together the helper utilities in the scripts/ directory
and third-party tools installed in the `scfv` conda environment to produce
clustered consensus scFv sequences from a merged read FASTQ dataset.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

SCRIPT_ROOT = Path(__file__).resolve().parent.parent


LINKER_SEQUENCE = "GGTGCTGGTGGCGGTAGCTGGAGGCGGTGGCTCTGGTGGTG"
UPSTREAM_TRIM = 360
DOWNSTREAM_TRIM = 360
TARGET_LENGTH = 700
DEFAULT_MIN_CLUSTER_SIZE = 30
ITERATION_CONFIG = [
    {"name": "cluster_only", "minid": None, "prefix": 80, "suffix": 80, "min_match": 60},
    {"name": "map95", "minid": 0.95, "prefix": 80, "suffix": 80, "min_match": 60},
    {"name": "map90", "minid": 0.90, "prefix": 80, "suffix": 80, "min_match": 55},
    {"name": "map85", "minid": 0.85, "prefix": 70, "suffix": 70, "min_match": 50},
    {"name": "map80", "minid": 0.80, "prefix": 60, "suffix": 60, "min_match": 45},
]


class PipelineError(RuntimeError):
    """Custom exception for pipeline failures."""


def log(message: str, log_file: Path | None = None) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    print(line)
    if log_file:
        with log_file.open("a") as fh:
            fh.write(line + "\n")


def run_command(
    cmd: List[str],
    log_file: Path,
    cwd: Path | None = None,
    capture_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    log(f"Running command: {' '.join(cmd)}", log_file)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=capture_output,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout if exc.stdout else ""
        stderr = exc.stderr if exc.stderr else ""
        error_msg = (
            f"Command failed with exit code {exc.returncode}: {' '.join(cmd)}\n"
            f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        )
        log(error_msg, log_file)
        raise PipelineError(error_msg) from exc
    return result


def read_fasta(path: Path) -> Dict[str, str]:
    sequences: Dict[str, str] = {}
    name: str | None = None
    seq_parts: List[str] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name:
                    sequences[name] = "".join(seq_parts)
                name = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if name:
            sequences[name] = "".join(seq_parts)
    return sequences


def write_fasta(path: Path, records: Dict[str, str]) -> None:
    with path.open("w") as fh:
        for name, seq in records.items():
            fh.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                fh.write(seq[i : i + 80] + "\n")


def merge_fastqs(primary: Path, secondary: Path, output: Path) -> Tuple[int, int]:
    """
    Merge FASTQ files while avoiding duplicate read IDs.
    Returns a tuple (total_reads, newly_added_reads).
    """
    seen: set[str] = set()
    added = 0
    total = 0

    def _write_record(out_handle, header, seq, plus, qual):
        out_handle.write(header)
        out_handle.write(seq)
        out_handle.write(plus)
        out_handle.write(qual)

    with output.open("w") as out:
        if primary.exists():
            with primary.open() as fh:
                while True:
                    header = fh.readline()
                    if not header:
                        break
                    seq = fh.readline()
                    plus = fh.readline()
                    qual = fh.readline()
                    rid = header.strip()[1:]
                    seen.add(rid)
                    total += 1
                    _write_record(out, header, seq, plus, qual)
        if secondary.exists():
            with secondary.open() as fh:
                while True:
                    header = fh.readline()
                    if not header:
                        break
                    seq = fh.readline()
                    plus = fh.readline()
                    qual = fh.readline()
                    rid = header.strip()[1:]
                    if rid in seen:
                        continue
                    seen.add(rid)
                    added += 1
                    total += 1
                    _write_record(out, header, seq, plus, qual)
    return total, added


def run_spoa(fastq_path: Path, output_path: Path, cluster_name: str, conda_env: str, log_file: Path) -> None:
    result = run_command(
        ["conda", "run", "-n", conda_env, "spoa", "-r", "0", str(fastq_path)],
        log_file,
        capture_output=True,
    )
    lines = result.stdout.strip().splitlines()
    sequence = "".join(line.strip() for line in lines if not line.startswith(">"))
    with output_path.open("w") as fh:
        fh.write(f">{cluster_name}\n")
        for i in range(0, len(sequence), 80):
            fh.write(sequence[i : i + 80] + "\n")


def run_seed_extend(
    consensus_path: Path,
    cluster_fastq: Path,
    merged_fastq: Path,
    output_dir: Path,
    prefix: int,
    suffix: int,
    min_match: int,
    log_file: Path,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    input_dir = output_dir / "input_fastq"
    input_dir.mkdir(exist_ok=True)
    consensus_records = read_fasta(consensus_path)
    if not consensus_records:
        raise PipelineError(f"No consensus sequence found in {consensus_path}")
    cluster_name = next(iter(consensus_records.keys()))
    temp_fastq = input_dir / f"{cluster_name}.fastq"
    shutil.copyfile(cluster_fastq, temp_fastq)
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "seed_extend.py"),
        "--consensus-dir",
        str(consensus_path),
        "--cluster-fastq-dir",
        str(input_dir),
        "--merged-fastq",
        str(merged_fastq),
        "--output-dir",
        str(output_dir),
        "--prefix-length",
        str(prefix),
        "--suffix-length",
        str(suffix),
        "--min-match",
        str(min_match),
    ]
    run_command(cmd, log_file)
    return output_dir / f"{cluster_name}_consensus.fa", output_dir / f"{cluster_name}.fastq"


def run_bbmap(
    ref: Path,
    reads: Path,
    outm: Path,
    minid: float,
    conda_env: str,
    log_file: Path,
    covstats: Path | None = None,
    workdir: Path | None = None,
) -> None:
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "bbmap.sh",
        f"ref={ref}",
        f"in={reads}",
        f"outm={outm}",
        "ambig=best",
        "maxindel=20",
        f"minid={minid}",
        "overwrite=t",
    ]
    if covstats:
        cmd.append(f"covstats={covstats}")
        # suppress SAM by redirecting to null
        cmd.append("out=stdout.sam")
    run_command(cmd, log_file, cwd=workdir)
    if covstats:
        # remove stdout.sam if created
        sam_path = (workdir or Path.cwd()) / "stdout.sam"
        if sam_path.exists():
            sam_path.unlink()


def orient_and_trim(
    seq: str,
    linker: str,
    upstream_trim: int,
    downstream_trim: int,
) -> Tuple[str, str, int, int, int, str]:
    linker_len = len(linker)
    rc = seq.translate(str.maketrans("ACGTacgt", "TGCAtgca"))[::-1]
    linker_pos = seq.find(linker)
    orientation = "forward"
    if linker_pos == -1:
        rc_pos = rc.find(linker)
        if rc_pos != -1:
            seq = rc
            linker_pos = rc_pos
            orientation = "reverse_complement"
        else:
            linker_pos = -1
            orientation = "unknown"
    if linker_pos >= 0:
        upstream_len = linker_pos
        downstream_len = len(seq) - (linker_pos + linker_len)
        start = max(0, linker_pos - upstream_trim)
        end = min(len(seq), linker_pos + linker_len + downstream_trim)
        trimmed = seq[start:end]
    else:
        upstream_len = len(seq)
        downstream_len = 0
        trimmed = seq
    trim_status = "complete"
    if linker_pos < 0 or len(trimmed) < TARGET_LENGTH - 40:
        trim_status = "partial"
    return seq, trimmed, linker_pos, upstream_len, downstream_len, trim_status


def best_identity(query: str, references: Dict[str, str]) -> Tuple[str, float]:
    compl = str.maketrans("ACGTacgt", "TGCAtgca")
    rc = query.translate(compl)[::-1]
    best_match = ("NA", 0.0)
    for ref_name, ref_seq in references.items():
        for candidate in (query, rc):
            if len(candidate) <= len(ref_seq):
                for start in range(len(ref_seq) - len(candidate) + 1):
                    window = ref_seq[start : start + len(candidate)]
                    matches = sum(a == b for a, b in zip(candidate, window))
                    identity = matches / len(candidate)
                    if identity > best_match[1]:
                        best_match = (ref_name, identity)
            else:
                for start in range(len(candidate) - len(ref_seq) + 1):
                    window = candidate[start : start + len(ref_seq)]
                    matches = sum(a == b for a, b in zip(window, ref_seq))
                    identity = matches / len(ref_seq)
                    if identity > best_match[1]:
                        best_match = (ref_name, identity)
    return best_match


def load_known_sequences(known_path: Path) -> Dict[str, str]:
    if not known_path.exists():
        raise PipelineError(f"Known scFv sequence file not found: {known_path}")
    return read_fasta(known_path)


def extend_cluster(
    cluster: str,
    initial_consensus: Path,
    initial_fastq: Path,
    merged_fastq: Path,
    base_dir: Path,
    conda_env: str,
    log_file: Path,
) -> Tuple[Path, Path, List[Dict[str, str]]]:
    cluster_dir = base_dir / cluster
    cluster_dir.mkdir(parents=True, exist_ok=True)
    current_consensus = initial_consensus
    current_fastq = cluster_dir / f"{cluster}_iteration0.fastq"
    shutil.copyfile(initial_fastq, current_fastq)
    history: List[Dict[str, str]] = []

    for idx, cfg in enumerate(ITERATION_CONFIG, start=1):
        iter_dir = cluster_dir / f"iter_{idx:02d}_{cfg['name']}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        seed_output_dir = iter_dir / "seed_extend"
        consensus_path, fastq_path = run_seed_extend(
            current_consensus,
            current_fastq,
            merged_fastq,
            seed_output_dir,
            cfg["prefix"],
            cfg["suffix"],
            cfg["min_match"],
            log_file,
        )

        iteration_record = {
            "iteration": cfg["name"],
            "seed_consensus": str(consensus_path),
            "seed_fastq": str(fastq_path),
        }

        if cfg["minid"] is not None:
            mapping_dir = iter_dir / "mapping"
            mapping_dir.mkdir(exist_ok=True)
            mapped_fastq = mapping_dir / f"{cluster}_mapped_min{int(cfg['minid']*100)}.fastq"
            run_bbmap(
                consensus_path,
                merged_fastq,
                mapped_fastq,
                cfg["minid"],
                conda_env,
                log_file,
                workdir=mapping_dir,
            )

            combined_fastq = iter_dir / f"{cluster}_combined.fastq"
            total, added = merge_fastqs(current_fastq, mapped_fastq, combined_fastq)
            iteration_record["mapped_reads"] = str(mapped_fastq)
            iteration_record["combined_fastq"] = str(combined_fastq)
            iteration_record["total_reads"] = str(total)
            iteration_record["new_reads"] = str(added)
            if mapped_fastq.exists():
                mapped_fastq.unlink()
            if added == 0:
                # No new information; keep existing consensus/fastq and stop iterating
                current_consensus = consensus_path
                current_fastq = current_fastq
                history.append(iteration_record)
                break
            current_fastq = combined_fastq
        else:
            combined_fastq = iter_dir / f"{cluster}_combined.fastq"
            total, _ = merge_fastqs(current_fastq, fastq_path, combined_fastq)
            iteration_record["combined_fastq"] = str(combined_fastq)
            iteration_record["total_reads"] = str(total)
            current_fastq = combined_fastq

        current_consensus = consensus_path
        seq = next(iter(read_fasta(consensus_path).values()))
        iteration_record["consensus_length"] = str(len(seq))
        history.append(iteration_record)
        if len(seq) >= TARGET_LENGTH:
            break

    return current_consensus, current_fastq, history


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the scFv reconstruction pipeline.")
    parser.add_argument("--input", required=True, type=Path, help="Merged FASTQ file (R1/R2 already merged).")
    parser.add_argument("--output", required=True, type=Path, help="Output directory for the pipeline run.")
    parser.add_argument("--conda-env", default="scfv", help="Conda environment name containing required tools.")
    parser.add_argument("--known-scfv", type=Path, default=None, help="FASTA file of known scFv sequences for comparison.")
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE, help="Minimum number of linker-positive reads per cluster.")
    parser.add_argument("--linker", default=LINKER_SEQUENCE, help="Linker sequence in forward orientation.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    input_fastq = args.input.resolve()
    if not input_fastq.exists():
        raise PipelineError(f"Input FASTQ not found: {input_fastq}")

    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "pipeline.log"
    log(f"Starting pipeline on {input_fastq}", log_file)

    extract_dir = output_dir / "01_linker_extract"
    cdr3_dir = output_dir / "02_cdr3_clustering"
    clusters_dir = output_dir / "03_cluster_reads"
    initial_consensus_dir = output_dir / "04_initial_consensus"
    extension_dir = output_dir / "05_extension"
    final_dir = output_dir / "final_results"
    for d in (extract_dir, cdr3_dir, clusters_dir, initial_consensus_dir, extension_dir, final_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: extract linker-oriented reads.
    extract_cmd = [
        sys.executable,
        str(Path(__file__).parent / "extract_linker_regions.py"),
        str(input_fastq),
        str(extract_dir),
        "--linker",
        args.linker,
        "--cdr3-offset",
        "120",
        "--vh-window",
        "160",
        "--cdr3-length",
        "60",
    ]
    run_command(extract_cmd, log_file)

    # Step 2: filter high-quality CDR3 sequences.
    raw_cdr3 = extract_dir / "vh_cdr3.fa"
    filtered_cdr3 = cdr3_dir / "vh_cdr3_highqual.fa"
    kept = 0
    total = 0
    with raw_cdr3.open() as inp, filtered_cdr3.open("w") as out:
        while True:
            header = inp.readline()
            if not header:
                break
            seq = inp.readline().strip()
            total += 1
            if seq.upper().count("N") <= 2:
                out.write(header)
                out.write(seq + "\n")
                kept += 1
    log(f"Filtered CDR3 segments: kept {kept} of {total}", log_file)

    # Step 3: vsearch clustering.
    cluster_centroids = cdr3_dir / "cdr3_centroids.fa"
    cluster_uc = cdr3_dir / "cdr3_assignments.uc"
    run_command(
        [
            "conda",
            "run",
            "-n",
            args.conda_env,
            "vsearch",
            "--cluster_fast",
            str(filtered_cdr3),
            "--id",
            "0.99",
            "--strand",
            "plus",
            "--qmask",
            "none",
            "--centroids",
            str(cluster_centroids),
            "--sizeout",
            "--uc",
            str(cluster_uc),
            "--fasta_width",
            "0",
        ],
        log_file,
    )

    # Step 4: partition reads by cluster.
    cluster_manifest = clusters_dir / "cluster_manifest.tsv"
    run_command(
        [
            sys.executable,
            str(Path(__file__).parent / "partition_by_cluster.py"),
            "--uc",
            str(cluster_uc),
            "--centroids",
            str(cluster_centroids),
            "--oriented-fastq",
            str(extract_dir / "linker_oriented.fastq"),
            "--output",
            str(clusters_dir),
            "--min-size",
            str(args.min_cluster_size),
            "--label-prefix",
            "cluster_",
        ],
        log_file,
    )
    if not cluster_manifest.exists():
        raise PipelineError("Cluster manifest not produced; check clustering output.")

    clusters: List[str] = []
    with cluster_manifest.open() as fh:
        next(fh)  # header
        for line in fh:
            parts = line.strip().split("\t")
            if len(parts) >= 1:
                cluster_name = parts[0]
                clusters.append(cluster_name)
    if not clusters:
        raise PipelineError("No clusters met the minimum size requirement.")
    log(f"Clusters to process: {', '.join(clusters)}", log_file)

    known_path = (args.known_scfv or (SCRIPT_ROOT / "reference" / "mock_truth_scfv.fa")).resolve()
    known_sequences = load_known_sequences(known_path)
    final_trimmed_records: Dict[str, str] = {}
    final_raw_records: Dict[str, str] = {}
    summary_rows: List[Dict[str, str]] = []

    for cluster in clusters:
        log(f"Processing cluster {cluster}", log_file)
        cluster_fastq = clusters_dir / f"{cluster}.fastq"
        if not cluster_fastq.exists():
            log(f"Skipping cluster {cluster}; FASTQ missing.", log_file)
            continue

        # Initial consensus via SPOA.
        initial_consensus = initial_consensus_dir / f"{cluster}.fa"
        run_spoa(cluster_fastq, initial_consensus, cluster, args.conda_env, log_file)

        # Extend cluster iteratively.
        final_consensus_path, final_fastq_path, history = extend_cluster(
            cluster,
            initial_consensus,
            cluster_fastq,
            input_fastq,
            extension_dir,
            args.conda_env,
            log_file,
        )

        raw_seq = next(iter(read_fasta(final_consensus_path).values()))
        oriented_seq, trimmed_seq, linker_pos, up_len, down_len, status = orient_and_trim(
            raw_seq,
            args.linker,
            UPSTREAM_TRIM,
            DOWNSTREAM_TRIM,
        )

        final_raw_records[cluster] = oriented_seq
        final_trimmed_records[cluster] = trimmed_seq

        # Save oriented consensus
        oriented_path = final_dir / f"{cluster}_consensus_raw.fa"
        write_fasta(oriented_path, {cluster: oriented_seq})
        trimmed_path = final_dir / f"{cluster}_consensus_trimmed.fa"
        write_fasta(trimmed_path, {cluster: trimmed_seq})

        # Map reads for coverage statistics
        coverage_dir = final_dir / "coverage"
        coverage_dir.mkdir(exist_ok=True)
        covstats_path = coverage_dir / f"{cluster}_covstats.txt"
        final_reads_dir = final_dir / "assigned_reads"
        final_reads_dir.mkdir(exist_ok=True)
        final_reads_path = final_reads_dir / f"{cluster}.fastq"
        shutil.copyfile(final_fastq_path, final_reads_path)
        mapped_cov_fastq = coverage_dir / f"{cluster}_mapped.fastq"
        run_bbmap(
            trimmed_path,
            final_reads_path,
            mapped_cov_fastq,
            0.95,
            args.conda_env,
            log_file,
            covstats=covstats_path,
            workdir=coverage_dir,
        )
        if mapped_cov_fastq.exists():
            mapped_cov_fastq.unlink()

        best_known, identity = best_identity(trimmed_seq, known_sequences)

        summary_rows.append(
            {
                "cluster": cluster,
                "raw_length": str(len(oriented_seq)),
                "trimmed_length": str(len(trimmed_seq)),
                "status": status,
                "orientation": "forward" if linker_pos >= 0 else "unknown",
                "linker_position": str(linker_pos),
                "upstream_length": str(up_len),
                "downstream_length": str(down_len),
                "best_known_match": best_known,
                "known_identity": f"{identity:.3f}",
                "final_reads": str(sum(1 for _ in open(final_reads_path)) // 4),
            }
        )
        log(
            f"Cluster {cluster}: raw length {len(oriented_seq)}, trimmed length {len(trimmed_seq)}, status {status}",
            log_file,
        )

    # Save combined FASTA files and summaries
    trimmed_fasta = final_dir / "scfv_consensus_trimmed.fa"
    raw_fasta = final_dir / "scfv_consensus_raw.fa"
    write_fasta(trimmed_fasta, final_trimmed_records)
    write_fasta(raw_fasta, final_raw_records)

    summary_path = final_dir / "qc_summary.tsv"
    with summary_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "cluster",
                "raw_length",
                "trimmed_length",
                "status",
                "orientation",
                "linker_position",
                "upstream_length",
                "downstream_length",
                "best_known_match",
                "known_identity",
                "final_reads",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # Copy known sequences alongside outputs
    shutil.copyfile(known_path, final_dir / "known_scfv.fa")

    log("Pipeline completed successfully.", log_file)
    log(f"Trimmed consensus sequences: {trimmed_fasta}", log_file)
    log(f"QC summary: {summary_path}", log_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
