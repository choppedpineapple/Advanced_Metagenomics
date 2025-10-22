#!/usr/bin/env python3
import sys
from collections import Counter

def read_fasta(path):
    seqs = []
    with open(path) as fh:
        name = None
        seq = []
        for line in fh:
            line=line.rstrip()
            if not line: 
                continue
            if line.startswith(">"):
                if name:
                    seqs.append((name, ''.join(seq)))
                name = line[1:].split()[0]
                seq = []
            else:
                seq.append(line.strip().upper())
        if name:
            seqs.append((name, ''.join(seq)))
    return seqs

def consensus_from_alignment(seqs):
    # seqs: list of (name, aligned_seq)
    if not seqs:
        return ""
    aln_len = max(len(s) for _, s in seqs)
    cons = []
    for i in range(aln_len):
        col = [s[i] for _, s in seqs if i < len(s) and s[i] not in ('-', 'N')]
        if not col:
            cons.append('N')   # no information -> N
        else:
            base = Counter(col).most_common(1)[0][0]
            cons.append(base)
    return ''.join(cons)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: consensus_from_alignment.py aligned_cluster.fasta", file=sys.stderr)
        sys.exit(1)

    aln_file = sys.argv[1]
    seqs = read_fasta(aln_file)
    cluster_size = len(seqs)
    consensus_seq = consensus_from_alignment(seqs)

    # create a descriptive header: use filename minus dirs
    import os
    name = os.path.basename(aln_file).replace("_aln.fasta","")
    print(f">{name}_n={cluster_size}")
    print(consensus_seq)
