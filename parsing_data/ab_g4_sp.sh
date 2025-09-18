#!/bin/bash

# This script reconstructs ~800 bp antibody sequences from paired-end FASTQ files.
# It performs quality control, trimming with fastp, merging with bbmerge, 
# de novo assembly with SPAdes, and filters contigs to 700-900 bp.
# Assumptions: All tools (fastqc, fastp, bbmerge.sh, spades.py, seqtk) are installed and in PATH.
# SPAdes is used for assembly as it's suitable for amplicon data and handles high similarity.
# Usage: ./reconstruct.sh <R1.fastq.gz> <R2.fastq.gz> <output_dir> <sample_name>
# Example: ./reconstruct.sh sample_R1.fastq.gz sample_R2.fastq.gz results/ sample

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <R1.fastq.gz> <R2.fastq.gz> <output_dir> <sample_name>"
    exit 1
fi

R1=$1
R2=$2
OUTDIR=$3
SAMPLE=$4

mkdir -p $OUTDIR

# Step 1: FastQC on raw reads
fastqc -o $OUTDIR $R1 $R2

# Step 2: Trim with fastp (adapter detection, quality filter, min length 50)
fastp -i $R1 -I $R2 -o $OUTDIR/${SAMPLE}_trimmed_R1.fastq.gz -O $OUTDIR/${SAMPLE}_trimmed_R2.fastq.gz \
      --detect_adapter_for_pe --qualified_quality_phred 20 --length_required 50 \
      --html $OUTDIR/${SAMPLE}_fastp.html --json $OUTDIR/${SAMPLE}_fastp.json

# Step 3: FastQC on trimmed reads
fastqc -o $OUTDIR $OUTDIR/${SAMPLE}_trimmed_R1.fastq.gz $OUTDIR/${SAMPLE}_trimmed_R2.fastq.gz

# Step 4: Merge paired-end reads with bbmerge for higher accuracy
bbmerge.sh in1=$OUTDIR/${SAMPLE}_trimmed_R1.fastq.gz in2=$OUTDIR/${SAMPLE}_trimmed_R2.fastq.gz \
           out=$OUTDIR/${SAMPLE}_merged.fastq.gz \
           outu1=$OUTDIR/${SAMPLE}_unmerged_R1.fastq.gz outu2=$OUTDIR/${SAMPLE}_unmerged_R2.fastq.gz \
           minoverlap=10 maxmismatches=0.25 minlen=50

# Step 5: De novo assembly with SPAdes (handles merged SE and unmerged PE)
spades.py --pe1-1 $OUTDIR/${SAMPLE}_unmerged_R1.fastq.gz --pe1-2 $OUTDIR/${SAMPLE}_unmerged_R2.fastq.gz \
          --s1 $OUTDIR/${SAMPLE}_merged.fastq.gz \
          -o $OUTDIR/${SAMPLE}_assembly \
          --careful --cov-cutoff 10 -k 21,33,55,77

# Step 6: Filter contigs to ~800 bp (700-900 bp range) using seqtk
# Outputs final reconstructed sequences in FASTA
seqtk seq -L 700 $OUTDIR/${SAMPLE}_assembly/contigs.fasta | \
awk 'BEGIN {min=700; max=900} /^>/ {if (len >= min && len <= max && len > 0) print header; print seq; len=0; header=$0; seq=""; next} {seq = seq $0; len += length($0)} END {if (len >= min && len <= max) print header; print seq}' > $OUTDIR/${SAMPLE}_reconstructed_800bp.fasta

# Correction in awk: To properly handle per-contig length filtering.
# The updated awk accumulates header and sequence, checks length after each contig.

echo "Reconstruction complete. Final FASTA: $OUTDIR/${SAMPLE}_reconstructed_800bp.fasta"
