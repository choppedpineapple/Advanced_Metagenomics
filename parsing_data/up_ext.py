#!/usr/bin/env python3
"""
extract_upstream_no_regex.py

Usage:
  python3 extract_upstream_no_regex.py input.fasta linker_seq up_len min_keep max_mismatch out_upstreams.fa

Example:
  python3 extract_upstream_no_regex.py merged.fa GGTG... 90 60 2 upstream_90.fa

Dependencies: biopython

What it does:
 - For each sequence, scans for forward linker or reverse-complement linker,
   allowing up to max_mismatch mismatches (Hamming).
 - Extracts up to up_len bases upstream of the first found linker instance.
 - Keeps only extracted subsequences with length >= min_keep.
 - Writes FASTA of kept upstream subsequences.
"""
import sys
from Bio import SeqIO
from Bio.Seq import Seq
from statistics import median

def revcomp(s):
    return str(Seq(s).reverse_complement())

def hamming_leq(a, b, max_mis):
    # assume same length
    mismatches = 0
    for x,y in zip(a,b):
        if x != y:
            mismatches += 1
            if mismatches > max_mis:
                return False, mismatches
    return True, mismatches

if len(sys.argv) != 7:
    print("Usage: python3 extract_upstream_no_regex.py input.fasta linker_seq up_len min_keep max_mismatch out_up.fa")
    sys.exit(1)

in_fasta = sys.argv[1]
linker = sys.argv[2].upper()
up_len = int(sys.argv[3])
min_keep = int(sys.argv[4])
max_mismatch = int(sys.argv[5])
out_fa = sys.argv[6]

linker_rc = revcomp(linker)
L = len(linker)

total = 0
found = 0
kept = 0
lengths = []
kept_records = []

for rec in SeqIO.parse(in_fasta, "fasta"):
    total += 1
    seq = str(rec.seq).upper()
    n = len(seq)
    matched = False

    # naive sliding window search for exact or up-to-max_mismatch positions
    # search forward
    for i in range(0, n - L + 1):
        window = seq[i:i+L]
        ok, mism = hamming_leq(window, linker, max_mismatch)
        if ok:
            # forward match at (i,i+L)
            start = max(0, i - up_len)
            upstream = seq[start:i]
            found += 1
            matched = True
            strand = '+'
            pos = i
            break
    if not matched:
        # search reverse complement
        for i in range(0, n - L + 1):
            window = seq[i:i+L]
            ok, mism = hamming_leq(window, linker_rc, max_mismatch)
            if ok:
                # reverse complement matched at seq[i:i+L]
                # for RC, the upstream relative to linkage on the original seq is the region after the match,
                # but to keep "VH upstream" direction consistently (5'->3'), we'll take the bases BEFORE the rc in rc orientation.
                # Simpler: get downstream region after match, then reverse-complement it.
                downstream = seq[i+L:i+L+up_len]
                upstream = str(Seq(downstream).reverse_complement())
                found += 1
                matched = True
                strand = '-'
                pos = i
                break

    if matched:
        # accept only if we have at least min_keep upstream bases
        if len(upstream) >= min_keep:
            kept += 1
            lengths.append(len(upstream))
            header = f"{rec.id}|pos={pos+1}|strand={strand}|ulen={len(upstream)}"
            kept_records.append((header, upstream))

# write FASTA
with open(out_fa, "w") as outfh:
    for h, s in kept_records:
        outfh.write(f">{h}\n")
        for i in range(0, len(s), 80):
            outfh.write(s[i:i+80] + "\n")

# summary
print("Input sequences:", total)
print("Sequences with linker-like match (<=%d mismatches): %d" % (max_mismatch, found))
print("Kept sequences (upstream >= %d): %d" % (min_keep, kept))
if lengths:
    print("Upstream length min/median/max: %d / %d / %d" % (min(lengths), int(median(lengths)), max(lengths)))
else:
    print("No upstream sequences passed the min_keep threshold.")
