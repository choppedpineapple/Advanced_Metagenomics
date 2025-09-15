#!/usr/bin/env python3
"""
FASTQ quality-trimmer
Keeps the 5'→3' portion of each read whose running mean Phred score
stays ≥ Q_THRESHOLD (default: 20).  Discards reads whose first base
already fails the threshold.

Usage:
    python fast_trim.py input.fastq output.fastq   # Q=20
    python fast_trim.py input.fastq output.fastq 25  # Q=25
Dependencies:
    pip install dnaio numpy
"""

import sys
import numpy as np
import dnaio                # C-backed, very fast FASTQ reader / writer

Q_THRESHOLD = 20             # default cut-off (Phred scale)

# ---------- helpers ---------------------------------------------------------- #
def parse_args() -> tuple[str, str, int]:
    if len(sys.argv) not in (3, 4):
        sys.exit(f"Usage: {sys.argv[0]} <in.fastq> <out.fastq> [Q]")
    q = int(sys.argv[3]) if len(sys.argv) == 4 else Q_THRESHOLD
    if q < 0:
        sys.exit("Quality threshold must be non-negative.")
    return sys.argv[1], sys.argv[2], q

def trim_front_mean(phred: np.ndarray, thr: int) -> int:
    """
    Return slice length to keep (0 if read is to be discarded).
    Vectorised cumulative mean keeps algorithm O(n) per read.
    """
    means = np.cumsum(phred, dtype=np.int32) / np.arange(1, phred.size + 1)
    bad_idx = np.flatnonzero(means < thr)
    return phred.size if bad_idx.size == 0 else bad_idx[0]

# ---------- main ------------------------------------------------------------- #
def main(in_fastq: str, out_fastq: str, q_cut: int):
    with dnaio.open(in_fastq, mode="r") as reader, \
         dnaio.open(out_fastq, mode="w") as writer:
        for rec in reader:
            # dnaio gives .qualities as ASCII str (Phred+33)
            phred = np.frombuffer(rec.qualities.encode("ascii"), dtype=np.uint8) - 33
            keep = trim_front_mean(phred, q_cut)
            if keep:   # skip if keep == 0
                writer.write(rec.name,
                             rec.sequence[:keep],
                             rec.qualities[:keep])

# ---------- entry-point ------------------------------------------------------ #
if __name__ == "__main__":
    main(*parse_args())
    
