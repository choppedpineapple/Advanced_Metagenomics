#!/bin/bash

# Set the project directory (change this to your actual project directory)
PROJECT_DIR="/path/to/your/project"

# Set number of threads (change according to your system)
THREADS=16  # Using 16 threads for faster processing

# Set reference database paths (you need to download these first)
KRAKEN2_DB="/path/to/kraken2/database"
BRACKEN_DB="/path/to/bracken/database"
HUMANN3_DB="/path/to/humann3/chocophlan/database"
HUMANN3_UTILS_DB="/path/to/humann3/utility/database"
METAPHLAN_DB="/path/to/metaphlan/database"

# Create output directories
mkdir -p "${PROJECT_DIR}/results/quality_control/raw"
mkdir -p "${PROJECT_DIR}/results/quality_control/trimmed"
mkdir -p "${PROJECT_DIR}/results/trimmed"
mkdir -p "${PROJECT_DIR}/results/classification"
mkdir -p "${PROJECT_DIR}/results/functional_profiling"
mkdir -p "${PROJECT_DIR}/results/logs"

# Function to process each sample
process_sample() {
    local SAMPLE_DIR=$1
    local SAMPLE_NAME=$(basename "${SAMPLE_DIR}")
    
    echo "[$(date +"%Y-%m-%d %T")] Processing sample: ${SAMPLE_NAME}"
    
    # Define input files
    R1="${SAMPLE_DIR}/${SAMPLE_NAME}_R1.fastq.gz"
    R2="${SAMPLE_DIR}/${SAMPLE_NAME}_R2.fastq.gz"
    
    # Check if input files exist
    if [ ! -f "${R1}" ] || [ ! -f "${R2}" ]; then
        echo "[$(date +"%Y-%m-%d %T")] Error: Missing input files for ${SAMPLE_NAME}"
        return 1
    fi
    
    # 1. Quality Control for raw data
    fastqc -t ${THREADS} "${R1}" "${R2}" -o "${PROJECT_DIR}/results/quality_control/raw" \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_fastqc_raw.log" 2>&1
    
    # 2. Quality Trimming
    TRIMMED_R1="${PROJECT_DIR}/results/trimmed/${SAMPLE_NAME}_R1_trimmed.fastq.gz"
    TRIMMED_R2="${PROJECT_DIR}/results/trimmed/${SAMPLE_NAME}_R2_trimmed.fastq.gz"
    TRIMMED_UNPAIRED_R1="${PROJECT_DIR}/results/trimmed/${SAMPLE_NAME}_R1_unpaired.fastq.gz"
    TRIMMED_UNPAIRED_R2="${PROJECT_DIR}/results/trimmed/${SAMPLE_NAME}_R2_unpaired.fastq.gz"
    
    trimmomatic PE -threads ${THREADS} \
        "${R1}" "${R2}" \
        "${TRIMMED_R1}" "${TRIMMED_UNPAIRED_R1}" \
        "${TRIMMED_R2}" "${TRIMMED_UNPAIRED_R2}" \
        ILLUMINACLIP:/path/to/adapters.fa:2:30:10 \
        LEADING:3 \
        TRAILING:3 \
        SLIDINGWINDOW:4:15 \
        MINLEN:50 \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_trimmomatic.log" 2>&1
    
    # Quality Control for trimmed data
    fastqc -t ${THREADS} "${TRIMMED_R1}" "${TRIMMED_R2}" -o "${PROJECT_DIR}/results/quality_control/trimmed" \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_fastqc_trimmed.log" 2>&1
    
    # 3. Taxonomic Classification with Kraken2
    KRAKEN_OUTPUT="${PROJECT_DIR}/results/classification/${SAMPLE_NAME}_kraken.out"
    KRAKEN_REPORT="${PROJECT_DIR}/results/classification/${SAMPLE_NAME}_report.txt"
    
    kraken2 --threads ${THREADS} \
        --db ${KRAKEN2_DB} \
        --paired \
        "${TRIMMED_R1}" "${TRIMMED_R2}" \
        --output "${KRAKEN_OUTPUT}" \
        --report "${KRAKEN_REPORT}" \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_kraken2.log" 2>&1
    
    # 4. Refinement of species-level abundance with Bracken
    BRACKEN_OUTPUT="${PROJECT_DIR}/results/classification/${SAMPLE_NAME}_bracken.txt"
    
    bracken -d ${KRAKEN2_DB} \
        -i "${KRAKEN_REPORT}" \
        -o "${BRACKEN_OUTPUT}" \
        -w "${PROJECT_DIR}/results/classification/${SAMPLE_NAME}_bracken_whole.txt" \
        -r 100 -l S \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_bracken.log" 2>&1
    
    # 5. Functional Profiling and Pathway Analysis with HUMAnN3
    HUMANN_OUTPUT_DIR="${PROJECT_DIR}/results/functional_profiling/${SAMPLE_NAME}"
    mkdir -p "${HUMANN_OUTPUT_DIR}"
    
    # First run MetaPhlAn to create the species abundance profile
    metaphlan "${TRIMMED_R1},${TRIMMED_R2}" \
        --input_type fastq \
        -o "${HUMANN_OUTPUT_DIR}/${SAMPLE_NAME}_metaphlan.txt" \
        --bowtie2db "${METAPHLAN_DB}" \
        --nproc ${THREADS} \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_metaphlan.log" 2>&1
    
    # Now run HUMAnN3 using the species abundance profile
    humann --input "${TRIMMED_R1},${TRIMMED_R2}" \
        --output "${HUMANN_OUTPUT_DIR}" \
        --threads ${THREADS} \
        --metaphlan-options="--input_type fastq" \
        --nucleotide-database "${HUMANN3_DB}" \
        --protein-database "${HUMANN3_UTILS_DB}" \
        --log "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_humann3.log" \
        --verbose
    
    # Create gene family and pathway summaries
    humann_join_tables -i "${HUMANN_OUTPUT_DIR}" \
        --file_name genefamilies \
        -o "${HUMANN_OUTPUT_DIR}/${SAMPLE_NAME}_genefamilies.tsv" \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_humann_join_gene.log" 2>&1
    
    humann_renorm_table -i "${HUMANN_OUTPUT_DIR}/${SAMPLE_NAME}_genefamilies.tsv" \
        --units cpm \
        -o "${HUMANN_OUTPUT_DIR}/${SAMPLE_NAME}_genefamilies_cpm.tsv" \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_humann_renorm_gene.log" 2>&1
    
    humann_join_tables -i "${HUMANN_OUTPUT_DIR}" \
        --file_name pathabundance \
        -o "${HUMANN_OUTPUT_DIR}/${SAMPLE_NAME}_pathabundance.tsv" \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_humann_join_path.log" 2>&1
    
    humann_renorm_table -i "${HUMANN_OUTPUT_DIR}/${SAMPLE_NAME}_pathabundance.tsv" \
        --units cpm \
        -o "${HUMANN_OUTPUT_DIR}/${SAMPLE_NAME}_pathabundance_cpm.tsv" \
        > "${PROJECT_DIR}/results/logs/${SAMPLE_NAME}_humann_renorm_path.log" 2>&1
    
    echo "[$(date +"%Y-%m-%d %T")] Completed processing sample: ${SAMPLE_NAME}"
}

# Generate MultiQC report
generate_multiqc_report() {
    echo "[$(date +"%Y-%m-%d %T")] Generating MultiQC report"
    multiqc "${PROJECT_DIR}/results/quality_control" \
        -o "${PROJECT_DIR}/results/quality_control" \
        > "${PROJECT_DIR}/results/logs/multiqc.log" 2>&1
}

# Export function so it's available to subshells
export -f process_sample
export -f generate_multiqc_report
export PROJECT_DIR
export THREADS
export KRAKEN2_DB
export BRACKEN_DB
export HUMANN3_DB
export HUMANN3_UTILS_DB
export METAPHLAN_DB

# Main processing loop
echo "[$(date +"%Y-%m-%d %T")] Starting metagenomics pipeline"

# Find all sample directories
SAMPLE_DIRS=("${PROJECT_DIR}"/*/)
    
# Process each sample in parallel (max 3 at a time)
for ((i=0; i<${#SAMPLE_DIRS[@]}; i+=3)); do
    for ((j=0; j<3 && i+j<${#SAMPLE_DIRS[@]}; j++)); do
        process_sample "${SAMPLE_DIRS[i+j]}" &
    done
    wait
done

# Generate final MultiQC report
generate_multiqc_report

echo "[$(date +"%Y-%m-%d %T")] Pipeline completed successfully"
