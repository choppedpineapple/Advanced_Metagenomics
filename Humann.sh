#!/bin/bash
set -euo pipefail  # Strict error handling

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --sample) SAMPLE="$2"; shift 2 ;;
    --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

R1="${OUTPUT_DIR}/merged_samples/${SAMPLE}_R1.fastq.gz"
R2="${OUTPUT_DIR}/merged_samples/${SAMPLE}_R2.fastq.gz"

# Step 1: Trim with fastp (exit on failure)
mkdir -p "${OUTPUT_DIR}/trimmed"
fastp \
  --in1 "${R1}" \
  --in2 "${R2}" \
  --out1 "${OUTPUT_DIR}/trimmed/${SAMPLE}_trimmed_R1.fastq.gz" \
  --out2 "${OUTPUT_DIR}/trimmed/${SAMPLE}_trimmed_R2.fastq.gz" \
  --json "${OUTPUT_DIR}/trimmed/${SAMPLE}_fastp.json" \
  --html "${OUTPUT_DIR}/trimmed/${SAMPLE}_fastp.html" \
  --thread "${SLURM_CPUS_PER_TASK}" || { echo "fastp failed"; exit 1; }

# Step 2: Host removal with Bowtie2 (exit on failure)
mkdir -p "${OUTPUT_DIR}/host_removed"
bowtie2 \
  -x /path/to/human_index \
  -1 "${OUTPUT_DIR}/trimmed/${SAMPLE}_trimmed_R1.fastq.gz" \
  -2 "${OUTPUT_DIR}/trimmed/${SAMPLE}_trimmed_R2.fastq.gz" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --un-conc-gz "${OUTPUT_DIR}/host_removed/${SAMPLE}_clean.fastq.gz" \
  > /dev/null || { echo "Bowtie2 failed"; exit 1; }

# Step 3: Concatenate R1/R2 for HUMAnN3 (no error suppression)
mkdir -p "${OUTPUT_DIR}/humann_input"
cat \
  "${OUTPUT_DIR}/host_removed/${SAMPLE}_clean.1.fastq.gz" \
  "${OUTPUT_DIR}/host_removed/${SAMPLE}_clean.2.fastq.gz" \
  > "${OUTPUT_DIR}/humann_input/${SAMPLE}_input.fastq.gz" || { echo "Concatenation failed"; exit 1; }

# Step 4: Run HUMAnN3 (exit on failure)
humann \
  --input "${OUTPUT_DIR}/humann_input/${SAMPLE}_input.fastq.gz" \
  --output "${OUTPUT_DIR}/humann_results" \
  --threads "${SLURM_CPUS_PER_TASK}" \
  --protein-database /path/to/uniref \
  --output-basename "${SAMPLE}" || { echo "HUMAnN3 failed"; exit 1; }

echo "Processing completed for ${SAMPLE}!"
