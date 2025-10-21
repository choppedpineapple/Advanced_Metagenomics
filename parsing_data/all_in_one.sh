#!/usr/bin/bash
# ===========================================================
# 1. CONVERT MERGED FASTQ TO FASTA
# ===========================================================
# Input: merged.fastq (from fastp + BBmerge)
# Output: merged.fasta
# -----------------------------------------------------------

seqtk seq -A merged.fastq > merged.fasta


# ===========================================================
# 2. FIND AND EXTRACT SEQUENCES CONTAINING LINKER (±2 mismatches)
# ===========================================================
# Input: merged.fasta
# Output: linker_hits.fasta
# -----------------------------------------------------------
# Define linker sequence (replace below with your actual 41bp linker)
LINKER="ACTGACTGACTGACTGACTGACTGACTGACTGACTGACTGA"

# Create reverse complement
LINKER_RC=$(echo "$LINKER" | tr "ATGCatgc" "TACGtacg" | rev)

# Locate the linker (≤2 mismatches)
seqkit locate -p "$LINKER" -m 2 merged.fasta > forward_linker_positions.txt
seqkit locate -p "$LINKER_RC" -m 2 merged.fasta > reverse_linker_positions.txt

# Extract IDs of sequences containing the linker or its reverse complement
awk 'NR>1 {print $1}' forward_linker_positions.txt > linker_ids.txt
awk 'NR>1 {print $1}' reverse_linker_positions.txt >> linker_ids.txt

# Remove duplicates
sort -u linker_ids.txt > linker_ids_unique.txt

# Extract the matching sequences into a new FASTA file
seqkit grep -f linker_ids_unique.txt merged.fasta > linker_hits.fasta


# ===========================================================
# 3. CLUSTER LINKER-CONTAINING SEQUENCES (97% SIMILARITY)
# ===========================================================
# Input: linker_hits.fasta
# Output: anchors_97.fa, anchors_97.uc
# -----------------------------------------------------------

vsearch --cluster_fast linker_hits.fasta \
        --id 0.97 \
        --centroids anchors_97.fa \
        --uc anchors_97.uc \
        --threads 8


# ===========================================================
# 4. MAP ALL ORIGINAL MERGED READS TO CLUSTER CENTROIDS
# ===========================================================
# Input: merged.fasta (all reads), anchors_97.fa (centroids)
# Output: clusters.uc, cluster_reads.fasta
# -----------------------------------------------------------

vsearch --usearch_global merged.fasta \
        --db anchors_97.fa \
        --id 0.97 \
        --uc clusters.uc \
        --matched cluster_reads.fasta \
        --threads 8


# ===========================================================
# 5. ASSEMBLE CLUSTERED READS WITH SPAdes
# ===========================================================
# Input: cluster_reads.fasta
# Output: spades_output/contigs.fasta
# -----------------------------------------------------------

spades.py --only-assembler \
          -s cluster_reads.fasta \
          -k 127 \
          -o spades_output


# ===========================================================
# 6. NOTES
# ===========================================================
# - anchors_97.fa: representative cluster centroids (anchors)
# - clusters.uc: cluster membership mapping
# - cluster_reads.fasta: recruited reads for assembly
# - spades_output/contigs.fasta: final assemblies
# -----------------------------------------------------------
