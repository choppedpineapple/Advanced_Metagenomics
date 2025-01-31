#!/bin/bash
#SBATCH --job-name=meta_pipeline
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --partition=bigmem
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G

# Pipeline Master Controller
# Submit this script first to initiate the workflow

CONFIG="pipeline_config.sh"
source ${CONFIG}

# Create job dependency chain
JOBID=$(sbatch --parsable \
    --job-name=meta_qc \
    --array=1-${NUM_SAMPLES}%${MAX_PARALLEL} \
    --output=logs/qc_%A_%a.out \
    --error=logs/qc_%A_%a.err \
    --partition=bigmem \
    --cpus-per-task=16 \
    --mem=64G \
    --time=4:00:00 \
    qc_trim.slurm)

JOBID=$(sbatch --parsable \
    --job-name=host_remove \
    --array=1-${NUM_SAMPLES}%${MAX_PARALLEL} \
    --output=logs/host_%A_%a.out \
    --error=logs/host_%A_%a.err \
    --partition=bigmem \
    --cpus-per-task=32 \
    --mem=128G \
    --time=6:00:00 \
    --dependency=afterok:${JOBID} \
    host_removal.slurm)

CLASS_JOBIDS=""
for CLASSIFIER in kraken2 centrifuge kaiju metaphlan motus; do
    CID=$(sbatch --parsable \
        --job-name=${CLASSIFIER} \
        --array=1-${NUM_SAMPLES}%${MAX_PARALLEL} \
        --output=logs/${CLASSIFIER}_%A_%a.out \
        --error=logs/${CLASSIFIER}_%A_%a.err \
        --partition=bigmem \
        --cpus-per-task=${CPUS_${CLASSIFIER}} \
        --mem=${MEM_${CLASSIFIER}} \
        --time=${TIME_${CLASSIFIER}} \
        --dependency=afterok:${JOBID} \
        classifiers/${CLASSIFIER}.slurm)
    CLASS_JOBIDS="${CLASS_JOBIDS}:${CID}"
done

# Consensus analysis depends on all classifiers
JOBID=$(sbatch --parsable \
    --job-name=consensus \
    --output=logs/consensus_%A.out \
    --error=logs/consensus_%A.err \
    --partition=bigmem \
    --cpus-per-task=48 \
    --mem=256G \
    --time=12:00:00 \
    --dependency=afterok${CLASS_JOBIDS} \
    consensus.slurm)

# Final visualization and reporting
sbatch \
    --job-name=viz \
    --output=logs/viz_%A.out \
    --error=logs/viz_%A.err \
    --partition=bigmem \
    --cpus-per-task=24 \
    --mem=64G \
    --time=2:00:00 \
    --dependency=afterok:${JOBID} \
    visualization.slurm
