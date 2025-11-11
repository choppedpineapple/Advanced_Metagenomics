#!/bin/bash
set -euo pipefail

############################
# CONFIG (edit these)
############################

# Inputs from previous step
UPSTREAM60="anchors.upstream_60.fasta"   # from extract_linker_with_upstream_v2.py
MERGED_FASTA="merged.fasta"              # all merged reads (FASTA)

# Optional: unmerged paired reads (if you want SPAdes to use pairs as well)
USE_PAIRED=false
R1_FASTQ="R1.clean.fastq.gz"
R2_FASTQ="R2.clean.fastq.gz"

# Clustering & recruitment thresholds
CLUSTER_ID=0.97      # identity for clustering upstream-60 anchors
RECRUIT_ID=0.90      # identity for recruiting reads to centroids (tune 0.85–0.95)

# How many clusters to assemble (by size)
TOP_N=100

# SPAdes
THREADS=12
KMERS="21,33,55,77,99,127"

# IgBlast (edit these paths to your sheep IMGT databases)
SHEEP_VH="imgt/sheep_IGHV.fasta"
SHEEP_DH="imgt/sheep_IGHD.fasta"
SHEEP_JH="imgt/sheep_IGHJ.fasta"
SHEEP_VL="imgt/sheep_IGKV_IGLV.fasta"   # use combined or pick one
SHEEP_JL="imgt/sheep_IGKJ_IGLJ.fasta"

# Expected scFv size window for filtering contigs
MIN_SCFV_LEN=600
MAX_SCFV_LEN=1000

# Output folders
WORKDIR="scfv_work"
ANCHOR_DIR="$WORKDIR/anchors97"
RECRUIT_DIR="$WORKDIR/recruit"
CLUSTERS_DIR="$WORKDIR/clusters_topN"
SPADES_DIR="$WORKDIR/spades_topN"
ANNOT_DIR="$WORKDIR/annot_topN"
SUMMARY_CSV="$WORKDIR/summary.csv"

mkdir -p "$WORKDIR" "$ANCHOR_DIR" "$RECRUIT_DIR" "$CLUSTERS_DIR" "$SPADES_DIR" "$ANNOT_DIR"

echo "=== Inputs ==="
echo "UPSTREAM60:     $UPSTREAM60"
echo "MERGED_FASTA:   $MERGED_FASTA"
echo "USE_PAIRED:     $USE_PAIRED"
$USE_PAIRED && echo "R1: $R1_FASTQ | R2: $R2_FASTQ"
echo


############################################
# 1) Cluster upstream-60 anchors at 97%
############################################
echo "[1/6] Clustering upstream-60 anchors at ${CLUSTER_ID}…"

vsearch --cluster_fast "$UPSTREAM60" \
        --id $CLUSTER_ID \
        --centroids "$ANCHOR_DIR/anchors_${CLUSTER_ID}.fa" \
        --uc "$ANCHOR_DIR/anchors_${CLUSTER_ID}.uc" \
        --threads $THREADS

# Get cluster sizes from UC
awk -F'\t' '($1=="H"||$1=="S"){counts[$2]++} END{for (c in counts) print counts[c], c}' \
    "$ANCHOR_DIR/anchors_${CLUSTER_ID}.uc" | sort -nr > "$ANCHOR_DIR/cluster_sizes.txt"

head -n 5 "$ANCHOR_DIR/cluster_sizes.txt" | sed 's/^/   /'
echo


###############################################################
# 2) Recruit full merged reads to *all* centroids at once
###############################################################
echo "[2/6] Recruiting merged reads to centroids at id ${RECRUIT_ID}…"

vsearch --usearch_global "$MERGED_FASTA" \
        --db "$ANCHOR_DIR/anchors_${CLUSTER_ID}.fa" \
        --id $RECRUIT_ID \
        --uc "$RECRUIT_DIR/recruited.uc" \
        --userout "$RECRUIT_DIR/recruited_hits.txt" \
        --userfields query+target+id+targetcov \
        --threads $THREADS

# Produce TOP_N cluster list by recruited count (query == centroid header)
awk '{print $1}' "$RECRUIT_DIR/recruited_hits.txt" | sort | uniq -c | sort -nr > "$RECRUIT_DIR/recruited_counts.txt"
cut -d' ' -f7- "$RECRUIT_DIR/recruited_counts.txt" | head -n $TOP_N > "$RECRUIT_DIR/topN_centroids.list"

echo "Top 5 centroids by recruited reads:"
head -n 5 "$RECRUIT_DIR/recruited_counts.txt" | sed 's/^/   /'
echo


######################################################################
# 3) Build per-cluster read ID lists & per-cluster FASTA (top N)
######################################################################
echo "[3/6] Building per-cluster FASTAs for TOP_N=$TOP_N …"

# Map UC cluster id -> centroid header
awk -F'\t' '($1=="S"){print $2"\t"$9}' "$ANCHOR_DIR/anchors_${CLUSTER_ID}.uc" > "$ANCHOR_DIR/clusterid_to_centroid.tsv"

# For each centroid header in topN, find its cluster id
> "$CLUSTERS_DIR/_topN_cluster_ids.tsv"
while read CENT; do
  CID=$(awk -v q="$CENT" '$2==q{print $1}' "$ANCHOR_DIR/clusterid_to_centroid.tsv" | head -n1)
  [ -n "$CID" ] && echo -e "$CID\t$CENT" >> "$CLUSTERS_DIR/_topN_cluster_ids.tsv"
done < "$RECRUIT_DIR/topN_centroids.list"

# Create per-cluster ID lists and FASTA
while read CID CENT; do
  echo "  - Cluster $CID  (centroid: $CENT)"
  awk -F'\t' -v q="$CENT" '($1=="H"||$1=="S"){ if($9==q) print $10 }' "$RECRUIT_DIR/recruited.uc" \
     | sort -u > "$CLUSTERS_DIR/cluster_${CID}_ids.txt"

  # include centroid itself just in case
  echo "$CENT" >> "$CLUSTERS_DIR/cluster_${CID}_ids.txt"
  sort -u "$CLUSTERS_DIR/cluster_${CID}_ids.txt" -o "$CLUSTERS_DIR/cluster_${CID}_ids.txt"

  # Extract full merged sequences
  seqtk subseq "$MERGED_FASTA" "$CLUSTERS_DIR/cluster_${CID}_ids.txt" > "$CLUSTERS_DIR/cluster_${CID}.fasta"
done < "$CLUSTERS_DIR/_topN_cluster_ids.tsv"

echo


############################################################
# 4) Assemble each cluster (SPAdes multi-k, careful)
############################################################
echo "[4/6] Assembling clusters with SPAdes…"

while read CID CENT; do
  INFA="$CLUSTERS_DIR/cluster_${CID}.fasta"
  OUTD="$SPADES_DIR/cluster_${CID}"
  mkdir -p "$OUTD"

  if [ "$USE_PAIRED" = true ]; then
    # Extract paired reads by ID if names match — OPTIONAL, uncomment if you want to include pairs
    # seqtk subseq "$R1_FASTQ" "$CLUSTERS_DIR/cluster_${CID}_ids.txt" > "$OUTD/cluster_${CID}_R1.fastq"
    # seqtk subseq "$R2_FASTQ" "$CLUSTERS_DIR/cluster_${CID}_ids.txt" > "$OUTD/cluster_${CID}_R2.fastq"
    # spades.py -1 "$OUTD/cluster_${CID}_R1.fastq" -2 "$OUTD/cluster_${CID}_R2.fastq" \
    #           -s "$INFA" -o "$OUTD" -k $KMERS --careful -t $THREADS 2> "$OUTD/spades.log"
    echo "    (paired assembly block commented out; enable if you extract R1/R2)"
  fi

  # Single-end (merged) assembly
  spades.py --only-assembler -s "$INFA" -o "$OUTD" -k $KMERS --careful -t $THREADS 2> "$OUTD/spades.log"

done < "$CLUSTERS_DIR/_topN_cluster_ids.tsv"

echo


############################################################################
# 5) Length filter contigs & run IgBlast (heavy + light) per cluster
############################################################################
echo "[5/6] Filtering contigs ${MIN_SCFV_LEN}-${MAX_SCFV_LEN} bp and running IgBlast…"

echo "cluster_id,contigs_total,contigs_in_window,longest_in_window" > "$SUMMARY_CSV"

while read CID CENT; do
  OUTD="$SPADES_DIR/cluster_${CID}"
  CONTIGS="$OUTD/contigs.fasta"
  if [ ! -s "$CONTIGS" ]; then
    echo "cluster_${CID},0,0,0" >> "$SUMMARY_CSV"
    continue
  fi

  # Count total contigs
  TOTAL=$(grep -c '^>' "$CONTIGS" || true)

  # Length-filter to expected scFv window
  FILTERED="$OUTD/contigs_${MIN_SCFV_LEN}_${MAX_SCFV_LEN}.fa"
  seqtk seq -L $MIN_SCFV_LEN -U $MAX_SCFV_LEN "$CONTIGS" > "$FILTERED" || true
  INWIN=$(grep -c '^>' "$FILTERED" || true)

  # Longest in window
  if [ -s "$FILTERED" ]; then
    LONGEST=$(awk '/^>/{if (seq_len){print seq_len}; seq_len=0; next} {seq_len+=length($0)} END{print seq_len}' "$FILTERED" | sort -nr | head -n1)
  else
    LONGEST=0
  fi

  # IgBlast Heavy
  if [ -s "$FILTERED" ]; then
    igblastn -query "$FILTERED" \
             -germline_db_V "$SHEEP_VH" \
             -germline_db_D "$SHEEP_DH" \
             -germline_db_J "$SHEEP_JH" \
             -organism sheep \
             -domain_system imgt \
             -outfmt "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore" \
             -out "$ANNOT_DIR/cluster_${CID}_IGH.tsv" 2> "$ANNOT_DIR/cluster_${CID}_IGH.log" || true

    # IgBlast Light
    igblastn -query "$FILTERED" \
             -germline_db_V "$SHEEP_VL" \
             -germline_db_J "$SHEEP_JL" \
             -organism sheep \
             -domain_system imgt \
             -outfmt "6 qseqid sseqid pident length qstart qend sstart send evalue bitscore" \
             -out "$ANNOT_DIR/cluster_${CID}_IGL.tsv" 2> "$ANNOT_DIR/cluster_${CID}_IGL.log" || true

    # Save filtered contigs too
    cp "$FILTERED" "$ANNOT_DIR/cluster_${CID}_contigs.fa"
  fi

  echo "cluster_${CID},$TOTAL,$INWIN,$LONGEST" >> "$SUMMARY_CSV"
done < "$CLUSTERS_DIR/_topN_cluster_ids.tsv"

echo


############################################
# 6) Done — show a tiny summary
############################################
echo "[6/6] Summary (top 10):"
column -s, -t "$SUMMARY_CSV" | head -n 11
echo
echo "Outputs:"
echo "  - Centroids:         $ANCHOR_DIR/anchors_${CLUSTER_ID}.fa"
echo "  - Recruit map:       $RECRUIT_DIR/recruited_hits.txt  (and recruited.uc)"
echo "  - Per-cluster FASTA: $CLUSTERS_DIR/cluster_<id>.fasta"
echo "  - SPAdes outputs:    $SPADES_DIR/cluster_<id>/contigs.fasta"
echo "  - IgBlast results:   $ANNOT_DIR/cluster_<id>_IGH.tsv / _IGL.tsv"
echo "  - Summary CSV:       $SUMMARY_CSV"
