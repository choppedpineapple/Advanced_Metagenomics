#!/bin/bash

# Set up environment and variables
THREADS=16  # Adjust based on your system
WORK_DIR="/path/to/your/working/directory"
INPUT_DIR="/path/to/input/fastq/files"
OUTPUT_DIR="/path/to/output/directory"

# Kraken2 and Bracken database paths
KRAKEN_DB="/path/to/kraken2/standard+human+plant+fungi+viral+archaea"
BRACKEN_DB="/path/to/bracken/database"

# Create output directories
mkdir -p ${OUTPUT_DIR}/fastp_output
mkdir -p ${OUTPUT_DIR}/kraken_output
mkdir -p ${OUTPUT_DIR}/bracken_output
mkdir -p ${OUTPUT_DIR}/abundance_plots
mkdir -p ${OUTPUT_DIR}/functional_analysis

# Function for error handling
error_exit() {
    echo "$1" >&2
    exit 1
}

# Preprocessing and Quality Control with fastp
# Note: This step includes quality trimming, adapter removal, and length filtering
preprocess_samples() {
    for R1 in ${INPUT_DIR}/*_R1_001.fastq.gz; do
        # Extract sample name
        SAMPLE=$(basename ${R1} _R1_001.fastq.gz)
        R2=${INPUT_DIR}/${SAMPLE}_R2_001.fastq.gz

        # Fastp preprocessing
        fastp \
            -i ${R1} \
            -I ${R2} \
            -o ${OUTPUT_DIR}/fastp_output/${SAMPLE}_R1_clean.fastq.gz \
            -O ${OUTPUT_DIR}/fastp_output/${SAMPLE}_R2_clean.fastq.gz \
            -h ${OUTPUT_DIR}/fastp_output/${SAMPLE}_fastp.html \
            -j ${OUTPUT_DIR}/fastp_output/${SAMPLE}_fastp.json \
            --length_required 50 \
            --correction \
            --cut_right \
            --thread ${THREADS} \
            || error_exit "Fastp preprocessing failed for ${SAMPLE}"
    done
}

# Kraken2 Taxonomic Classification
# De-hosting is done during Kraken2 classification to remove human/plant reads
kraken2_classification() {
    for R1 in ${OUTPUT_DIR}/fastp_output/*_R1_clean.fastq.gz; do
        SAMPLE=$(basename ${R1} _R1_clean.fastq.gz)
        R2=${OUTPUT_DIR}/fastp_output/${SAMPLE}_R2_clean.fastq.gz

        # Kraken2 classification with paired-end reads
        kraken2 \
            --db ${KRAKEN_DB} \
            --paired \
            --threads ${THREADS} \
            --output ${OUTPUT_DIR}/kraken_output/${SAMPLE}_kraken.output \
            --report ${OUTPUT_DIR}/kraken_output/${SAMPLE}_kraken.report \
            --classified-out ${OUTPUT_DIR}/kraken_output/${SAMPLE}_classified#.fastq \
            ${R1} ${R2} \
            || error_exit "Kraken2 classification failed for ${SAMPLE}"
    done
}

# Bracken Abundance Estimation
bracken_abundance() {
    for REPORT in ${OUTPUT_DIR}/kraken_output/*_kraken.report; do
        SAMPLE=$(basename ${REPORT} _kraken.report)
        
        # Run Bracken for species-level abundance
        bracken \
            -d ${KRAKEN_DB} \
            -i ${REPORT} \
            -o ${OUTPUT_DIR}/bracken_output/${SAMPLE}_bracken_species.txt \
            -w ${OUTPUT_DIR}/bracken_output/${SAMPLE}_bracken_species_report.txt \
            -r 150 \
            -l S \
            || error_exit "Bracken abundance estimation failed for ${SAMPLE}"
        
        # Run Bracken for genus-level abundance
        bracken \
            -d ${KRAKEN_DB} \
            -i ${REPORT} \
            -o ${OUTPUT_DIR}/bracken_output/${SAMPLE}_bracken_genus.txt \
            -w ${OUTPUT_DIR}/bracken_output/${SAMPLE}_bracken_genus_report.txt \
            -r 150 \
            -l G \
            || error_exit "Bracken abundance estimation failed for ${SAMPLE}"
    done
}

# Krakentools for Visualization and Manipulation
krakentools_analysis() {
    # Convert Kraken reports to Krona format for interactive visualization
    for REPORT in ${OUTPUT_DIR}/kraken_output/*_kraken.report; do
        ktImportTaxonomy \
            -o ${OUTPUT_DIR}/abundance_plots/$(basename ${REPORT} _kraken.report)_krona.html \
            ${REPORT} \
            || error_exit "Krona visualization failed"
    done

    # Extract classified reads
    kreport2mpa.py \
        -r ${OUTPUT_DIR}/kraken_output/*_kraken.report \
        -o ${OUTPUT_DIR}/abundance_plots/mpa_style_report.txt \
        || error_exit "MPA-style report generation failed"
}

# Functional Analysis with HUMAnN3
# Note: We'll use HUMAnN3 with MetaPhlAn4 for taxonomy, but focus on functional analysis
functional_analysis() {
    for R1 in ${OUTPUT_DIR}/fastp_output/*_R1_clean.fastq.gz; do
        SAMPLE=$(basename ${R1} _R1_clean.fastq.gz)
        R2=${OUTPUT_DIR}/fastp_output/${SAMPLE}_R2_clean.fastq.gz

        # HUMAnN3 functional profiling
        humann3 \
            --input ${R1} \
            --output ${OUTPUT_DIR}/functional_analysis/${SAMPLE}_humann3 \
            --threads ${THREADS} \
            || error_exit "HUMAnN3 functional analysis failed for ${SAMPLE}"
    done
}

# Diversity Analysis
diversity_analysis() {
    # Use R or Python script for diversity analysis
    # This is a placeholder - you'll need to implement specific diversity calculations
    python3 << EOF
import pandas as pd
import numpy as np
from scipy.stats import entropy

def calculate_shannon_diversity(abundance_file):
    # Read Bracken species abundance file
    df = pd.read_csv(abundance_file, sep='\t')
    
    # Calculate Shannon diversity
    proportions = df['fraction_total_reads'].values
    shannon = entropy(proportions)
    
    return shannon

# Process all Bracken species abundance files
diversity_results = {}
for abundance_file in glob.glob('${OUTPUT_DIR}/bracken_output/*_bracken_species.txt'):
    sample_name = os.path.basename(abundance_file).replace('_bracken_species.txt', '')
    diversity_results[sample_name] = calculate_shannon_diversity(abundance_file)

# Save diversity results
pd.DataFrame.from_dict(diversity_results, orient='index', columns=['Shannon_Diversity']) \
    .to_csv('${OUTPUT_DIR}/diversity_analysis.csv')
EOF
}

# Main Pipeline Execution
main() {
    preprocess_samples
    kraken2_classification
    bracken_abundance
    krakentools_analysis
    functional_analysis
    diversity_analysis
}

# Run the pipeline
main

echo "Microbiome Analysis Pipeline Completed Successfully!"
