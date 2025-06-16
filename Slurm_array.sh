#!/bin/bash
#SBATCH --job-name=kraken2_array
#SBATCH --array=0-126
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/kraken2_%A_%a.out
#SBATCH --error=logs/kraken2_%A_%a.err

# Load Kraken2 module (adjust if you're using conda or a custom path)
module load kraken2

# Get the sample name for this task
SAMPLE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" sample_names.txt)

# Define input/output
R1="test-samples/${SAMPLE}_R1.fastq.gz"
R2="test-samples/${SAMPLE}_R2.fastq.gz"
OUTDIR="kraken2_output"
mkdir -p $OUTDIR

# Run Kraken2
kraken2 --paired \
  --db /path/to/kraken2-db \
  --threads 4 \
  --report $OUTDIR/${SAMPLE}_report.txt \
  --output $OUTDIR/${SAMPLE}_output.txt \
  $R1 $R2

-----------

NUM_SAMPLES=$(wc -l < sample_names.txt)

sbatch --array=0-$(($(wc -l < sample_names.txt)-1)) kraken2_array.slurm
