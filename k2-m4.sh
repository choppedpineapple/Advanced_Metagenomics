#!/bin/bash

# Comprehensive Metagenomics Profiling Script
# For paired-end Illumina shotgun metagenomics data
# Usage: bash metagenomics_profiling.sh <forward_read> <reverse_read>

# Exit on any error
set -e

# Check if correct number of arguments are provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <forward_read.fastq.gz> <reverse_read.fastq.gz>"
    exit 1
fi

# Input files
READ1=$1
READ2=$2

# Output directory
OUTPUT_DIR="metagenomics_analysis_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

# Number of threads (adjust based on your system)
THREADS=$(nproc)

# Log file
LOG_FILE="${OUTPUT_DIR}/analysis_log.txt"

# Start logging
echo "Metagenomics Analysis Started: $(date)" > $LOG_FILE

# 1. Quality Control with FastQC
echo "Running FastQC for quality control..." | tee -a $LOG_FILE
mkdir -p ${OUTPUT_DIR}/fastqc_output
fastqc -t $THREADS -o ${OUTPUT_DIR}/fastqc_output $READ1 $READ2 >> $LOG_FILE 2>&1

# 2. Optional: Adapter and Quality Trimming with Trimmomatic
echo "Performing adapter and quality trimming..." | tee -a $LOG_FILE
mkdir -p ${OUTPUT_DIR}/trimmed
trimmomatic PE -threads $THREADS \
    $READ1 $READ2 \
    ${OUTPUT_DIR}/trimmed/paired_forward.fastq.gz \
    ${OUTPUT_DIR}/trimmed/unpaired_forward.fastq.gz \
    ${OUTPUT_DIR}/trimmed/paired_reverse.fastq.gz \
    ${OUTPUT_DIR}/trimmed/unpaired_reverse.fastq.gz \
    ILLUMINACLIP:/path/to/adapters/TruSeq3-PE.fa:2:30:10 \
    LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36 >> $LOG_FILE 2>&1

# Set trimmed reads as input for downstream analysis
TRIM_READ1="${OUTPUT_DIR}/trimmed/paired_forward.fastq.gz"
TRIM_READ2="${OUTPUT_DIR}/trimmed/paired_reverse.fastq.gz"

# 3. Kraken2 Taxonomic Classification
echo "Running Kraken2 taxonomic classification..." | tee -a $LOG_FILE
mkdir -p ${OUTPUT_DIR}/kraken2_output

# Ensure you have downloaded the appropriate Kraken2 database
KRAKEN_DB="/path/to/kraken2/standard_database"

kraken2 --db $KRAKEN_DB \
    --paired \
    --threads $THREADS \
    --report ${OUTPUT_DIR}/kraken2_output/kraken2_report.txt \
    --output ${OUTPUT_DIR}/kraken2_output/kraken2_classification.txt \
    --gzip-compressed \
    $TRIM_READ1 $TRIM_READ2 >> $LOG_FILE 2>&1

# 4. Kraken2 Visualization with Krona
echo "Generating Krona visualization from Kraken2 results..." | tee -a $LOG_FILE
ktImportTaxonomy ${OUTPUT_DIR}/kraken2_output/kraken2_classification.txt \
    -o ${OUTPUT_DIR}/kraken2_output/krona_visualization.html >> $LOG_FILE 2>&1

# 5. MetaPhlAn4 Taxonomic Profiling
echo "Running MetaPhlAn4 taxonomic profiling..." | tee -a $LOG_FILE
mkdir -p ${OUTPUT_DIR}/metaphlan4_output

# Perform MetaPhlAn4 analysis
metaphlan $TRIM_READ1,$TRIM_READ2 \
    --input_type fastq \
    --nproc $THREADS \
    --bowtie2db /path/to/metaphlan/database \
    --output_file ${OUTPUT_DIR}/metaphlan4_output/metaphlan4_profile.txt \
    --samout ${OUTPUT_DIR}/metaphlan4_output/alignment.sam.bz2 \
    --tax_lev a \  # Analyze at all taxonomic levels
    --unclassified_estimation \
    --use_large_index >> $LOG_FILE 2>&1

# 6. Generate MetaPhlAn4 Visualization
echo "Creating MetaPhlAn4 visualization..." | tee -a $LOG_FILE
metaphlan ${OUTPUT_DIR}/metaphlan4_output/metaphlan4_profile.txt \
    --input_type profile \
    --output_file ${OUTPUT_DIR}/metaphlan4_output/metaphlan4_plot.png \
    --plot_type barplot >> $LOG_FILE 2>&1

# 7. Compare Kraken2 and MetaPhlAn4 Results
echo "Comparing Kraken2 and MetaPhlAn4 results..." | tee -a $LOG_FILE
# You may want to add custom comparison scripts here

# Final log
echo "Metagenomics Analysis Completed: $(date)" >> $LOG_FILE

# Print final summary
echo "Analysis complete. Results are in $OUTPUT_DIR"
cat $LOG_FILE

# Optional: Clean up intermediate files
# Uncomment if you want to save disk space
# rm -rf ${OUTPUT_DIR}/trimmed
