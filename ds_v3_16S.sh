#!/bin/bash
# 16S Functional Metagenomics Pipeline for HPC
# Usage: ./16s_pipeline.sh -1 forward_reads.fastq.gz -2 reverse_reads.fastq.gz -o output_dir -t threads

# Parse command-line arguments
while getopts "1:2:o:t:" opt; do
  case $opt in
    1) R1="$OPTARG" ;;
    2) R2="$OPTARG" ;;
    o) OUTDIR="$OPTARG" ;;
    t) THREADS="$OPTARG" ;;
    *) echo "Usage: $0 -1 forward_reads.fastq.gz -2 reverse_reads.fastq.gz -o output_dir -t threads" >&2
       exit 1 ;;
  esac
done

# Check required arguments
if [ -z "$R1" ] || [ -z "$R2" ] || [ -z "$OUTDIR" ] || [ -z "$THREADS" ]; then
  echo "Missing required arguments!" >&2
  echo "Usage: $0 -1 forward_reads.fastq.gz -2 reverse_reads.fastq.gz -o output_dir -t threads" >&2
  exit 1
fi

# Check if input files exist
if [ ! -f "$R1" ] || [ ! -f "$R2" ]; then
  echo "Input files do not exist!" >&2
  exit 1
fi

# Create output directory structure
mkdir -p "$OUTDIR"/{fastqc_raw,trimmed,fastqc_trimmed,merged,quality_filtered,chimeras_removed,taxonomy,functional_analysis}

# Load necessary modules (modify according to your HPC environment)
module load fastqc
module load cutadapt
module load vsearch
module load qiime2/2023.5  # Or your preferred version
module load picard
module load kraken2
module load humann

# Function to check command success
check_success() {
  if [ $? -ne 0 ]; then
    echo "Error in step: $1" >&2
    exit 1
  fi
}

# Step 1: Quality control of raw reads
echo "Step 1/10: Running FastQC on raw reads..."
fastqc -t $THREADS -o "$OUTDIR"/fastqc_raw "$R1" "$R2"
check_success "FastQC raw"

# Step 2: Read trimming and adapter removal
echo "Step 2/10: Trimming reads with Cutadapt..."
cutadapt -a AGATCGGAAGAGCACACGTCTGAACTCCAGTCA \
         -A AGATCGGAAGAGCGTCGTGTAGGGAAAGAGTGT \
         -q 20 --minimum-length 100 --max-n 0 \
         -o "$OUTDIR"/trimmed/trimmed_R1.fastq.gz \
         -p "$OUTDIR"/trimmed/trimmed_R2.fastq.gz \
         -j $THREADS "$R1" "$R2" > "$OUTDIR"/trimmed/trimming_report.txt
check_success "Cutadapt"

# Step 3: Quality control of trimmed reads
echo "Step 3/10: Running FastQC on trimmed reads..."
fastqc -t $THREADS -o "$OUTDIR"/fastqc_trimmed \
       "$OUTDIR"/trimmed/trimmed_R1.fastq.gz \
       "$OUTDIR"/trimmed/trimmed_R2.fastq.gz
check_success "FastQC trimmed"

# Step 4: Read merging (for overlapping paired-end reads)
echo "Step 4/10: Merging paired-end reads with VSEARCH..."
vsearch --fastq_mergepairs "$OUTDIR"/trimmed/trimmed_R1.fastq.gz \
        --reverse "$OUTDIR"/trimmed/trimmed_R2.fastq.gz \
        --fastqout "$OUTDIR"/merged/merged.fastq.gz \
        --fastq_minovlen 20 \
        --fastq_maxdiffs 5 \
        --threads $THREADS \
        --fastq_allowmergestagger \
        --fastqout_notmerged_fwd "$OUTDIR"/merged/unmerged_R1.fastq.gz \
        --fastqout_notmerged_rev "$OUTDIR"/merged/unmerged_R2.fastq.gz
check_success "VSEARCH merge"

# Step 5: Quality filtering
echo "Step 5/10: Quality filtering with VSEARCH..."
vsearch --fastq_filter "$OUTDIR"/merged/merged.fastq.gz \
        --fastq_maxee 1.0 \
        --fastq_minlen 200 \
        --fastq_maxlen 500 \
        --fastq_maxns 0 \
        --fastaout "$OUTDIR"/quality_filtered/filtered.fasta \
        --fastaout_discarded "$OUTDIR"/quality_filtered/discarded.fasta \
        --threads $THREADS
check_success "VSEARCH quality filter"

# Step 6: Dereplication
echo "Step 6/10: Dereplicating sequences..."
vsearch --derep_fulllength "$OUTDIR"/quality_filtered/filtered.fasta \
        --output "$OUTDIR"/quality_filtered/dereplicated.fasta \
        --sizeout \
        --relabel "ASV_" \
        --threads $THREADS
check_success "VSEARCH dereplication"

# Step 7: Chimera removal
echo "Step 7/10: Removing chimeras with VSEARCH..."
vsearch --uchime3_denovo "$OUTDIR"/quality_filtered/dereplicated.fasta \
        --nonchimeras "$OUTDIR"/chimeras_removed/nonchimeras.fasta \
        --chimeras "$OUTDIR"/chimeras_removed/chimeras.fasta \
        --threads $THREADS
check_success "VSEARCH chimera removal"

# Step 8: Clustering into ASVs (Amplicon Sequence Variants)
echo "Step 8/10: Clustering ASVs with VSEARCH..."
vsearch --cluster_size "$OUTDIR"/chimeras_removed/nonchimeras.fasta \
        --id 0.97 \
        --centroids "$OUTDIR"/chimeras_removed/ASVs.fasta \
        --sizein --sizeout \
        --relabel "ASV_" \
        --threads $THREADS
check_success "VSEARCH ASV clustering"

# Step 9: Taxonomic classification
echo "Step 9/10: Taxonomic classification with Kraken2..."
# Note: You need to download the appropriate Kraken2 database first
kraken2 --db /path/to/kraken2_db \
        --threads $THREADS \
        --output "$OUTDIR"/taxonomy/kraken2_output.txt \
        --report "$OUTDIR"/taxonomy/kraken2_report.txt \
        --use-names \
        --paired \
        --gzip-compressed \
        "$OUTDIR"/trimmed/trimmed_R1.fastq.gz \
        "$OUTDIR"/trimmed/trimmed_R2.fastq.gz
check_success "Kraken2 classification"

# Step 10: Functional analysis (PICRUSt2 or similar)
echo "Step 10/10: Functional prediction with PICRUSt2..."
# Note: You need to have PICRUSt2 installed and configured
picrust2_pipeline.py -i "$OUTDIR"/chimeras_removed/ASVs.fasta \
                     -o "$OUTDIR"/functional_analysis \
                     -p $THREADS \
                     --in_traits EC,KO,Metacyc \
                     --verbose
check_success "PICRUSt2"

# Generate final report
echo "Pipeline completed successfully!"
echo "Results are in: $OUTDIR"
echo "Quality control reports: $OUTDIR/fastqc_raw and $OUTDIR/fastqc_trimmed"
echo "Processed sequences: $OUTDIR/quality_filtered"
echo "ASVs: $OUTDIR/chimeras_removed/ASVs.fasta"
echo "Taxonomic classification: $OUTDIR/taxonomy"
echo "Functional analysis: $OUTDIR/functional_analysis"
