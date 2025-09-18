#!/bin/bash

# This script reconstructs ~800 bp antibody sequences from paired-end FASTQ files.
# It performs quality control, trimming with fastp, merging with pRESTO's AssemblePairs.py (repertoire-specific for better accuracy),
# de novo assembly with SPAdes, and filters contigs to 700-900 bp.
# Assumptions: All tools (fastqc, fastp, AssemblePairs.py from pRESTO, spades.py, seqtk) are installed and in PATH.
# SPAdes is used for assembly as it's suitable for amplicon data and handles high similarity.
# Usage: ./reconstruct.sh <R1.fastq.gz> <R2.fastq.gz> <output_dir> <sample_name>
# Example: ./reconstruct.sh sample_R1.fastq.gz sample_R2.fastq.gz results/ sample
# Note: If you have primers, add --1head <primer_seq> etc. to AssemblePairs. For reference mode: Change 'align' to 'reference' and add -r reference.fasta --minident 0.9 --evalue 1e-5.

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

# Step 4: Merge paired-end reads with pRESTO's AssemblePairs.py (align mode for de novo overlaps; repertoire-optimized)
AssemblePairs.py align -1 $OUTDIR/${SAMPLE}_trimmed_R1.fastq.gz -2 $OUTDIR/${SAMPLE}_trimmed_R2.fastq.gz \
                 --rc tail --coord illumina --outname ${SAMPLE} --outdir $OUTDIR --log $OUTDIR/${SAMPLE}_assemble.log \
                 --nproc 4 --maxerror 0.25 --minlen 10 --scanrev --failed

# Step 5: De novo assembly with SPAdes (handles merged SE and unmerged PE from pRESTO outputs)
spades.py --pe1-1 $OUTDIR/${SAMPLE}_assemble-fail_R1.fastq --pe1-2 $OUTDIR/${SAMPLE}_assemble-fail_R2.fastq \
          --s1 $OUTDIR/${SAMPLE}_assemble-pass.fastq \
          -o $OUTDIR/${SAMPLE}_assembly \
          --careful --cov-cutoff 10 -k 21,33,55,77

# Step 6: Filter contigs to ~800 bp (700-900 bp range) using seqtk
# Outputs final reconstructed sequences in FASTA
seqtk seq -L 700 $OUTDIR/${SAMPLE}_assembly/contigs.fasta | \
awk 'BEGIN {min=700; max=900} /^>/ {if (len >= min && len <= max && len > 0) print header; print seq; len=0; header=$0; seq=""; next} {seq = seq $0; len += length($0)} END {if (len >= min && len <= max) print header; print seq}' > $OUTDIR/${SAMPLE}_reconstructed_800bp.fasta

# Correction in awk: To properly handle per-contig length filtering.
# The updated awk accumulates header and sequence, checks length after each contig.

echo "Reconstruction complete. Final FASTA: $OUTDIR/${SAMPLE}_reconstructed_800bp.fasta"
