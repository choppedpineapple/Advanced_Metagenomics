seqkit grep -s -i -r -p GGTGGAGGCGGTTCAGGCGGAGGTGGAGG merged.fastq > linker_reads.fastq

seqkit locate -p GGTGGAGGCGGTTCAGGCGGAGGTGGAGG linker_reads.fastq \
| awk '{start=$2-90; if(start<1) start=1; print $1"\t"start"\t"$2}' > upstream_regions.bed

seqkit subseq -f upstream_regions.bed linker_reads.fastq > upstream_90bp.fa

vsearch --cluster_fast upstream_90bp.fa --id 0.9 --centroids clonotypes.fa


spades.py -s cluster1_reads.fastq -k 127 -o cluster1_assembly

-----------------

# 1. Convert fastq → fasta if not already
seqtk seq -a merged.fastq > merged.fasta

# 2. Find linker matches and save coordinates
seqkit locate -p "GGTGGTGGTGGTAGC" -i merged.fasta > locate.tsv

# 3. Extract 90 bases upstream of each linker occurrence
seqkit subseq -f locate.tsv -u 90 merged.fasta > upstream_90.fa


----------------++

#!/usr/bin/env python3
"""
extract_upstream_fuzzy.py

Usage:
  python3 extract_upstream_fuzzy.py merged.fasta GGTGGTGGTGGTAGC 90 upstream_90.fa

Find approximate occurrences of a linker (allowing mismatches)
and extract up to N bases upstream of the first match in each read.

Requires: pip install biopython regex
"""

import sys, regex
from Bio import SeqIO

if len(sys.argv) != 5:
    print("Usage: python3 extract_upstream_fuzzy.py <in.fasta> <linker> <up_len> <out.fasta>")
    sys.exit(1)

in_fa, linker, up_len, out_fa = sys.argv[1:]
linker = linker.upper()
up_len = int(up_len)

# Build fuzzy regex pattern (allow up to 2 mismatches)
pattern = f"({linker}){{e<=2}}"

records_out = []
count = 0

for rec in SeqIO.parse(in_fa, "fasta"):
    seq = str(rec.seq).upper()
    m = regex.search(pattern, seq)
    if m:
        start = max(0, m.start() - up_len)
        subseq = seq[start:m.start()]
        if subseq:
            count += 1
            header = f"{rec.id}|linker_pos={m.start()+1}|mismatches={m.fuzzy_counts}"
            records_out.append((header, subseq))

with open(out_fa, "w") as f:
    for h, s in records_out:
        f.write(f">{h}\n")
        for i in range(0, len(s), 80):
            f.write(s[i:i+80] + "\n")

print(f"[+] Extracted {count} upstream sequences → {out_fa}")


-----

python3 extract_upstream_fuzzy.py merged.fasta GGTGGTGGTGGTAGC 90 upstream_90.fa
