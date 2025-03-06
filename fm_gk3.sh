#!/bin/bash

# Set variables
INPUT_DIR="raw_fastq"           # Directory with input fastq.gz files
OUTPUT_DIR="pipeline_output"    # Output directory
THREADS=16                      # Multi-threading (adjust based on your system)
KRAKEN_DB="/path/to/kraken2_db" # Kraken2 database (e.g., Standard or PlusPFP)
HUMAN_DB="/path/to/human_db"    # Human genome database for kneaddata (e.g., GRCh38)
HUMANN_DB="/path/to/humann_db"  # HUMAnN3 database (e.g., UniRef90 + MetaCyc)

# Create output directories
mkdir -p ${OUTPUT_DIR}/fastp ${OUTPUT_DIR}/kneaddata ${OUTPUT_DIR}/kraken2 ${OUTPUT_DIR}/bracken ${OUTPUT_DIR}/humann3 ${OUTPUT_DIR}/plots

# Step 1: Quality control and trimming with fastp
echo "Running fastp for quality control and trimming..."
for R1 in ${INPUT_DIR}/*_R1_001.fastq.gz; do
    R2=${R1/_R1_/_R2_}
    SAMPLE=$(basename ${R1} _R1_001.fastq.gz)
    fastp \
        -i ${R1} -I ${R2} \
        -o ${OUTPUT_DIR}/fastp/${SAMPLE}_R1.fastq.gz \
        -O ${OUTPUT_DIR}/fastp/${SAMPLE}_R2.fastq.gz \
        --detect_adapter_for_pe \
        --thread ${THREADS} \
        --html ${OUTPUT_DIR}/fastp/${SAMPLE}_fastp.html \
        --json ${OUTPUT_DIR}/fastp/${SAMPLE}_fastp.json
done

# Step 2: De-hosting with kneaddata
echo "Running kneaddata for human read removal..."
for R1 in ${OUTPUT_DIR}/fastp/*_R1.fastq.gz; do
    R2=${R1/_R1/_R2_}
    SAMPLE=$(basename ${R1} _R1.fastq.gz)
    kneaddata \
        -i ${R1} -i ${R2} \
        -o ${OUTPUT_DIR}/kneaddata \
        -db ${HUMAN_DB} \
        --trimmomatic-options "SLIDINGWINDOW:4:20 MINLEN:50" \
        --bowtie2-options "--very-sensitive" \
        --threads ${THREADS} \
        --output-prefix ${SAMPLE}
done

# Step 3: Kraken2 taxonomic classification
echo "Running Kraken2 for taxonomic classification..."
for R1 in ${OUTPUT_DIR}/kneaddata/*_R1_kneaddata_paired_1.fastq; do
    R2=${R1/_1.fastq/_2.fastq}
    SAMPLE=$(basename ${R1} _R1_kneaddata_paired_1.fastq)
    kraken2 \
        --db ${KRAKEN_DB} \
        --threads ${THREADS} \
        --paired ${R1} ${R2} \
        --output ${OUTPUT_DIR}/kraken2/${SAMPLE}.kraken2 \
        --report ${OUTPUT_DIR}/kraken2/${SAMPLE}_report.txt
done

# Step 4: Bracken for abundance estimation
echo "Running Bracken for abundance estimation..."
for REPORT in ${OUTPUT_DIR}/kraken2/*_report.txt; do
    SAMPLE=$(basename ${REPORT} _report.txt)
    bracken \
        -d ${KRAKEN_DB} \
        -i ${REPORT} \
        -o ${OUTPUT_DIR}/bracken/${SAMPLE}_bracken.txt \
        -r 150 \ # Read length (adjust if different)
        -l S \   # Species level
        -t 10    # Minimum read threshold
done

# Step 5: Filter out plant reads (Viridiplantae, taxid 33090)
echo "Filtering out plant reads from Bracken output..."
for BRACKEN in ${OUTPUT_DIR}/bracken/*_bracken.txt; do
    SAMPLE=$(basename ${BRACKEN} _bracken.txt)
    awk '$2 != 33090 && $2 != 0' ${BRACKEN} > ${OUTPUT_DIR}/bracken/${SAMPLE}_filtered_bracken.txt
done

# Step 6: Calculate Shannon diversity with Krakentools
echo "Calculating Shannon diversity..."
for BRACKEN in ${OUTPUT_DIR}/bracken/*_filtered_bracken.txt; do
    SAMPLE=$(basename ${BRACKEN} _filtered_bracken.txt)
    kreport2krona.py \
        -r ${BRACKEN} \
        -o ${OUTPUT_DIR}/bracken/${SAMPLE}_krona.txt
    # Shannon diversity (requires abundance column, adjust as needed)
    python3 -c "import sys; import math; abund = [float(line.split('\t')[5]) for line in open(sys.argv[1]) if float(line.split('\t')[5]) > 0]; total = sum(abund); p = [x/total for x in abund]; shannon = -sum(x * math.log(x) for x in p); print(f'Shannon diversity for ${SAMPLE}: {shannon}')" ${BRACKEN} > ${OUTPUT_DIR}/bracken/${SAMPLE}_shannon.txt
done

# Step 7: Functional analysis with HUMAnN3
echo "Running HUMAnN3 for functional profiling..."
for R1 in ${OUTPUT_DIR}/kneaddata/*_R1_kneaddata_paired_1.fastq; do
    R2=${R1/_1.fastq/_2.fastq}
    SAMPLE=$(basename ${R1} _R1_kneaddata_paired_1.fastq)
    cat ${R1} ${R2} > ${OUTPUT_DIR}/kneaddata/${SAMPLE}_combined.fastq
    humann3 \
        --input ${OUTPUT_DIR}/kneaddata/${SAMPLE}_combined.fastq \
        --output ${OUTPUT_DIR}/humann3 \
        --threads ${THREADS} \
        --taxonomic-profile ${OUTPUT_DIR}/bracken/${SAMPLE}_filtered_bracken.txt \
        --nucleotide-database ${HUMANN_DB}/chocophlan \
        --protein-database ${HUMANN_DB}/uniref
done

# Step 8: Visualization
echo "Generating taxonomic and pathway plots..."
# Taxonomy plot (Krona)
for KRONA in ${OUTPUT_DIR}/bracken/*_krona.txt; do
    SAMPLE=$(basename ${KRONA} _krona.txt)
    krona \
        -o ${OUTPUT_DIR}/plots/${SAMPLE}_taxonomy.html \
        ${KRONA}
done

# Normalize HUMAnN3 output and generate pathway plots
humann_renorm_table \
    --input ${OUTPUT_DIR}/humann3/pathabundance.tsv \
    --output ${OUTPUT_DIR}/humann3/pathabundance_relab.tsv \
    --units relab
humann_barplot \
    --input ${OUTPUT_DIR}/humann3/pathabundance_relab.tsv \
    --output ${OUTPUT_DIR}/plots/pathway_abundance.pdf

echo "Pipeline completed!"
