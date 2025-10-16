#!/usr/bin/env python3
"""
extract_upstream_strict_filter.py

Usage:
  python3 extract_upstream_strict_filter.py merged.fasta LINKER UP_LEN MIN_KEEP out.fa

- merged.fasta : merged reads (FASTA)
- LINKER       : linker DNA sequence (exact match, case-insensitive)
- UP_LEN       : number of bases to extract upstream of first match attempt
- MIN_KEEP     : minimum upstream length to keep (discard smaller)
- out.fa       : output fasta of kept upstream sequences

This script:
- searches for LINKER (exact) and its reverse complement
- for the first match per read, extracts up to UP_LEN bases upstream (or whatever is available)
- discards result if upstream length < MIN_KEEP
- writes summary stats (counts, lengths)
"""
import sys
from Bio import SeqIO

if len(sys.argv) != 6:
    print("Usage: python3 extract_upstream_strict_filter.py merged.fasta LINKER UP_LEN MIN_KEEP out.fa")
    sys.exit(1)

in_fa = sys.argv[1]
linker = sys.argv[2].upper()
up_len = int(sys.argv[3])
min_keep = int(sys.argv[4])
out_fa = sys.argv[5]

def revcomp(s):
    trans = str.maketrans("ACGTN", "TGCAN")
    return s.translate(trans)[::-1]

linker_rc = revcomp(linker)

kept = []
lengths = []
total = 0
found = 0

for rec in SeqIO.parse(in_fa, "fasta"):
    total += 1
    seq = str(rec.seq).upper()
    # find first exact linker or rc occurrence
    idx = seq.find(linker)
    strand = '+'
    if idx == -1:
        idx = seq.find(linker_rc)
        strand = '-'
    if idx != -1:
        found += 1
        start = max(0, idx - up_len)
        subseq = seq[start:idx]  # upstream (not including linker)
        L = len(subseq)
        if L >= min_keep:
            header = f"{rec.id}|linker_pos={idx+1}|strand={strand}|up_len={L}"
            kept.append((header, subseq))
            lengths.append(L)

# write output fasta
with open(out_fa, "w") as fh:
    for h,s in kept:
        fh.write(f">{h}\n")
        for i in range(0, len(s), 80):
            fh.write(s[i:i+80] + "\n")

# print summary
import statistics
print(f"Input reads: {total}")
print(f"Reads with exact-linker (or RC) found: {found}")
print(f"Kept reads (up_len >= {min_keep}): {len(kept)}")
if lengths:
    print(f"Upstream length (min/median/mean/max): {min(lengths)}/{int(statistics.median(lengths))}/{statistics.mean(lengths):.1f}/{max(lengths)}")
else:
    print("No sequences kept (lengths empty).")

print(f"Wrote {len(kept)} sequences to {out_fa}")
