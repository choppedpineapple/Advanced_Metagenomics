#!/usr/bin/env python3
"""
illumina_qc.py

A learning-focused script for processing Illumina FASTQ (gzipped) reads
using Biopython. Supports single-end (SE) and paired-end (PE) data. Modular
functions cover:

- End trimming based on per-base quality threshold.
- Sliding-window trimming based on average quality.
- Adapter trimming by simple substring search.
- Read filtering by minimum length or average quality.
- Basic QC metrics: GC content, average quality.

Run as a script to process files, or import and call individual functions.
"""

import argparse
import gzip
from Bio import SeqIO

# -------------------------------
# Trimming & Filtering Functions
# -------------------------------

def trim_low_quality_end(record, threshold):
    """
    Trim low-quality bases from 3' end of a SeqRecord.
    Keeps trimming until the last base has quality >= threshold.
    """
    quals = record.letter_annotations["phred_quality"]
    cut_pos = len(quals)
    while cut_pos > 0 and quals[cut_pos - 1] < threshold:
        cut_pos -= 1
    return record[:cut_pos]


def trim_sliding_window(record, window_size, threshold):
    """
    Trim the read from the first window (of size window_size) whose
    average quality < threshold. All bases from that point onward
    are removed.
    """
    quals = record.letter_annotations["phred_quality"]
    n = len(quals)
    for i in range(n - window_size + 1):
        window_avg = sum(quals[i:i + window_size]) / window_size
        if window_avg < threshold:
            return record[:i]
    return record  # no trimming needed


def trim_adapter(record, adapter_seq):
    """
    Trim adapter_seq (string) from the read if found.
    Trims from the start of the adapter onward.
    """
    seq_str = str(record.seq)
    idx = seq_str.find(adapter_seq)
    if idx != -1:
        return record[:idx]
    return record


def filter_by_length(record, min_length):
    """
    Return the record if its length >= min_length, else return None.
    """
    return record if len(record.seq) >= min_length else None


def calculate_average_quality(record):
    """
    Compute average Phred quality of the read.
    """
    quals = record.letter_annotations["phred_quality"]
    return sum(quals) / len(quals) if quals else 0.0


def filter_by_average_quality(record, threshold):
    """
    Return the record if its average quality >= threshold, else None.
    """
    return record if calculate_average_quality(record) >= threshold else None


def calculate_gc_content(record):
    """
    Compute GC-content (0-100%) of the read.
    """
    seq = str(record.seq).upper()
    gc = seq.count('G') + seq.count('C')
    return 100.0 * gc / len(seq) if seq else 0.0


# -------------------------------
# Processing Pipelines
# -------------------------------

def process_single_end(in_path, out_path, qc_params):
    """
    Process single-end gzipped FASTQ.
    qc_params is a dict with keys:
      - qual_trim
      - window_size
      - window_qual
      - min_length
      - adapter (or None)
    """
    with gzip.open(in_path, "rt") as hin, gzip.open(out_path, "wt") as hout:
        for record in SeqIO.parse(hin, "fastq"):
            # 1) Adapter trimming
            if qc_params["adapter"]:
                record = trim_adapter(record, qc_params["adapter"])
            # 2) End trimming
            record = trim_low_quality_end(record, qc_params["qual_trim"])
            # 3) Sliding-window trimming
            record = trim_sliding_window(record,
                                         qc_params["window_size"],
                                         qc_params["window_qual"])
            # 4) Filter by length
            record = filter_by_length(record, qc_params["min_length"])
            if record:
                SeqIO.write(record, hout, "fastq")


def process_paired_end(r1_in, r2_in, r1_out, r2_out, qc_params):
    """
    Process paired-end gzipped FASTQ.
    Applies identical trimming to both mates and writes only pairs
    where both reads pass filters.
    """
    with gzip.open(r1_in, "rt") as h1, gzip.open(r2_in, "rt") as h2, \
         gzip.open(r1_out, "wt") as out1, gzip.open(r2_out, "wt") as out2:
        for rec1, rec2 in zip(SeqIO.parse(h1, "fastq"),
                               SeqIO.parse(h2, "fastq")):
            # Process mate 1
            if qc_params["adapter"]:
                rec1 = trim_adapter(rec1, qc_params["adapter"])
            rec1 = trim_low_quality_end(rec1, qc_params["qual_trim"])
            rec1 = trim_sliding_window(rec1,
                                       qc_params["window_size"],
                                       qc_params["window_qual"])
            rec1 = filter_by_length(rec1, qc_params["min_length"])
            # Process mate 2
            if qc_params["adapter"]:
                rec2 = trim_adapter(rec2, qc_params["adapter"])
            rec2 = trim_low_quality_end(rec2, qc_params["qual_trim"])
            rec2 = trim_sliding_window(rec2,
                                       qc_params["window_size"],
                                       qc_params["window_qual"])
            rec2 = filter_by_length(rec2, qc_params["min_length"])
            # Write only if both mates survive
            if rec1 and rec2:
                SeqIO.write(rec1, out1, "fastq")
                SeqIO.write(rec2, out2, "fastq")


# -------------------------------
# Command-line Interface
# -------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Illumina FASTQ QC: trimming and filtering of gzipped reads."
    )
    p.add_argument("--mode", choices=["SE", "PE"], required=True,
                   help="Single-end (SE) or paired-end (PE).")
    p.add_argument("--in1", required=True,
                   help="Input FASTQ.gz (SE) or mate1 (PE).")
    p.add_argument("--out1", required=True,
                   help="Output FASTQ.gz (SE) or mate1 (PE).")
    p.add_argument("--in2", help="Input mate2 FASTQ.gz (for PE).")
    p.add_argument("--out2", help="Output mate2 FASTQ.gz (for PE).")
    p.add_argument("--qual_trim", type=int, default=20,
                   help="Quality threshold for end trimming (default: 20).")
    p.add_argument("--window_size", type=int, default=4,
                   help="Sliding window size (default: 4).")
    p.add_argument("--window_qual", type=int, default=20,
                   help="Average quality threshold for sliding window (default: 20).")
    p.add_argument("--min_length", type=int, default=50,
                   help="Minimum read length to keep (default: 50).")
    p.add_argument("--adapter", default=None,
                   help="Adapter sequence to trim (optional).")
    return p.parse_args()


def main():
    args = parse_args()
    qc_params = {
        "qual_trim": args.qual_trim,
        "window_size": args.window_size,
        "window_qual": args.window_qual,
        "min_length": args.min_length,
        "adapter": args.adapter
    }

    if args.mode == "SE":
        process_single_end(args.in1, args.out1, qc_params)
    else:
        if not args.in2 or not args.out2:
            raise ValueError("PE mode requires --in2 and --out2")
        process_paired_end(args.in1, args.in2, args.out1, args.out2, qc_params)


if __name__ == "__main__":
    main()
