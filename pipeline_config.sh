#!/bin/bash
# SLURM Pipeline Configuration

# Cluster Parameters
export MAX_PARALLEL=20  # Max concurrent array jobs
export NUM_SAMPLES=$(ls ${INPUT_DIR}/*_R1.fastq.gz | wc -l)

# Resource Allocations
declare -A CPUS=(
    [kraken2]=32
    [centrifuge]=40
    [kaiju]=64
    [metaphlan]=24
    [motus]=32
)

declare -A MEM=(
    [kraken2]="128G"
    [centrifuge]="200G"
    [kaiju]="96G"
    [metaphlan]="64G"
    [motus]="160G"
)

declare -A TIME=(
    [kraken2]="6:00:00"
    [centrifuge]="8:00:00"
    [kaiju]="12:00:00"
    [metaphlan]="4:00:00"
    [motus]="6:00:00"
)

# Path Configurations
export INPUT_DIR="/path/to/raw_data"
export OUTPUT_DIR="/path/to/results_$(date +%Y%m%d)"
export REFERENCE_DB="/shared/databases"
export HOST_REF="/shared/hg38"
export TMPDIR="/lscratch/${SLURM_JOB_ID}"
