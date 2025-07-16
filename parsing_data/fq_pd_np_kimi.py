#!/usr/bin/env python3
"""
fastq2df.py – Parse FASTQ (plain or gzipped) and store data in pandas + numpy.

Usage:
    python fastq2df.py sample.fastq[.gz]
"""

import sys
import gzip
from pathlib import Path
from Bio import SeqIO
import pandas as pd
import numpy as np

# ------------------------------------------------------------------
# 1. Handle command-line argument
# ------------------------------------------------------------------
if len(sys.argv) != 2:
    sys.exit("Usage: python fastq2df.py <fastq_file>")

fastq_path = Path(sys.argv[1])

# ------------------------------------------------------------------
# 2. Open file transparently (plain or gzipped)
# ------------------------------------------------------------------
def open_fastq(path: Path):
    if path.suffix.lower() == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")

# ------------------------------------------------------------------
# 3. Parse FASTQ and fill lists
# ------------------------------------------------------------------
headers, seqs, quals, lengths = [], [], [], []

with open_fastq(fastq_path) as handle:
    for rec in SeqIO.parse(handle, "fastq"):
        headers.append(rec.id)
        seqs.append(str(rec.seq))
        quals.append("".join(chr(q + 33) for q in rec.letter_annotations["phred_quality"]))
        lengths.append(len(rec))

# ------------------------------------------------------------------
# 4. Build pandas DataFrame
# ------------------------------------------------------------------
df = pd.DataFrame({
    "header": headers,
    "sequence": seqs,
    "quality": quals,
    "length": lengths
})

print("Pandas DataFrame (head):")
print(df.head())

# ------------------------------------------------------------------
# 5. Build numpy structured array (recarray)
# ------------------------------------------------------------------
max_len = max(lengths)
# Fixed-length string dtypes chosen to be at least as long as the longest entry
dtype = np.dtype([
    ("header",   f"U{max(len(h) for h in headers)}"),
    ("sequence", f"U{max(lengths)}"),
    ("quality",  f"U{max(lengths)}"),
    ("length",   "i4")
])

rec_arr = np.rec.fromarrays([headers, seqs, quals, lengths], dtype=dtype)

print("\nNumPy recarray (head):")
print(rec_arr[:3])

# ------------------------------------------------------------------
# 6. Optional: save outputs
# ------------------------------------------------------------------
# df.to_csv("fastq_data.csv", index=False)
# np.save("fastq_data.npy", rec_arr)
