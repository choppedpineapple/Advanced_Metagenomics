vsearch --cluster_fast input.fasta --id 0.90 --centroids centroids.fasta --clusters cluster_

mkdir -p aligned_consensus
OUTPUT="all_consensus.fasta"
> $OUTPUT

for f in cluster_*.fasta; do
  # skip empty clusters
  [ -s "$f" ] || continue

  out_aln="aligned_consensus/${f%.fasta}_aln.fasta"
  muscle -in "$f" -out "$out_aln" -quiet

  # call per-cluster consensus (python script below)
  python3 consensus_from_alignment.py "$out_aln" >> "$OUTPUT"
done

echo "Wrote all consensus sequences to $OUTPUT"
