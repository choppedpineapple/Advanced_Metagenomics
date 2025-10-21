#!/usr/bin/env python3
from collections import Counter

input_file = "nucleotide_sequences.txt"
output_file = "consensus_sequences.fasta"

# --- Parameters ---
MIN_CLUSTER_IDENTITY = 0.97  # adjust this if you want looser/tighter clustering

# --- Read all sequences ---
with open(input_file) as f:
    seqs = [line.strip().upper() for line in f if line.strip()]

# --- Basic clustering by sequence similarity ---
# Note: simple all-vs-all identity-based grouping (no external tools)
def seq_identity(s1, s2):
    l = min(len(s1), len(s2))
    matches = sum(a == b for a, b in zip(s1[:l], s2[:l]))
    return matches / l if l > 0 else 0

clusters = []
for seq in seqs:
    found = False
    for cluster in clusters:
        rep = cluster[0]
        if seq_identity(seq, rep) >= MIN_CLUSTER_IDENTITY:
            cluster.append(seq)
            found = True
            break
    if not found:
        clusters.append([seq])

# --- Consensus calling ---
def consensus(sequences):
    max_len = max(len(s) for s in sequences)
    consensus_seq = ""
    for i in range(max_len):
        bases = [s[i] for s in sequences if i < len(s)]
        base = Counter(bases).most_common(1)[0][0]
        consensus_seq += base
    return consensus_seq

# --- Write output ---
with open(output_file, "w") as out:
    for i, cluster in enumerate(clusters, 1):
        cons = consensus(cluster)
        out.write(f">cluster_{i}_n={len(cluster)}\n{cons}\n")

print(f"✅ Done! {len(clusters)} consensus sequences written to {output_file}")
