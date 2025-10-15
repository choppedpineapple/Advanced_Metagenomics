#!/usr/bin/env python3
"""
extract_upstream_linker.py

Usage:
  python3 extract_upstream_linker.py merged.fasta LINKER 90 upstream_90.fa

Find occurrences of LINKER in sequences (case-insensitive).
For each sequence that contains the LINKER, extract up to N bases upstream
(of the first occurrence) and write to a FASTA file.

This works on fasta. If you have fastq use seqtk to convert first:
  seqtk seq -a merged.fastq > merged.fasta
"""
import sys
from Bio import SeqIO

if len(sys.argv) != 5:
    print("Usage: python3 extract_upstream_linker.py merged.fasta LINKER UP_LEN out.fa")
    sys.exit(1)

in_fasta = sys.argv[1]
linker = sys.argv[2].upper()
up_len = int(sys.argv[3])
out_fa = sys.argv[4]

out = []
count = 0
for rec in SeqIO.parse(in_fasta, "fasta"):
    seq = str(rec.seq).upper()
    idx = seq.find(linker)
    if idx != -1:
        start = max(0, idx - up_len)
        subseq = seq[start:idx]  # upstream region (not including linker)
        if len(subseq) > 0:
            count += 1
            header = f"{rec.id}|linker_pos={idx+1}"
            out.append((header, subseq))

with open(out_fa, "w") as fh:
    for h,s in out:
        fh.write(f">{h}\n")
        # wrap at 80 chars
        for i in range(0,len(s),80):
            fh.write(s[i:i+80]+"\n")

print(f"Wrote {len(out)} upstream sequences to {out_fa}")
