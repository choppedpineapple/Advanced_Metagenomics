# get cluster sizes & cluster ids sorted (uc format)
awk '/^C/ {print $2, $9}' anchors_97.uc | awk '{counts[$1]++} END{for (k in counts) print counts[k], k}' | sort -nr > cluster_sizes.txt

# get top 50 clusters
head -n 50 cluster_sizes.txt | awk '{print $2}' > top_clusters.txt

# for each cluster id produce a list of member read ids
mkdir -p cluster_members
for cid in $(cat top_clusters.txt); do
  # cluster lines start with 'S' (seed) or 'H' (hit). field 2 is cluster id, field 9 is sequence label in uc format for vsearch.
  awk -v id="$cid" -F'\t' '$2==id {print $9}' anchors_97.uc > cluster_members/cluster_${cid}_ids.txt
done

# now extract full sequences (original fasta) for each cluster
# Replace original_full.fasta with your initial FASTA that contains the complete reads (merged reads)
for f in cluster_members/cluster_*_ids.txt; do
  outname=${f%.txt}.fasta
  seqtk subseq /path/to/original_full.fasta $f > $outname
done
