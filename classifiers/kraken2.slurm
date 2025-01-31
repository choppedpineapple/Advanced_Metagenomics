#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=${CPUS_kraken2}
#SBATCH --mem=${MEM_kraken2}

sample=$(ls ${INPUT_DIR}/*_R1.fastq.gz | sed -n ${SLURM_ARRAY_TASK_ID}p | xargs basename)
base=${sample%_R1.fastq.gz}

module load kraken2/2.1.2 bracken/2.8

kraken2 --paired \
    --db ${REFERENCE_DB}/kraken2_stdplus \
    --threads ${SLURM_CPUS_PER_TASK} \
    --confidence 0.1 \
    --minimum-hit-groups 3 \
    --report ${OUTPUT_DIR}/kraken2/${base}_report.txt \
    --output - \
    ${OUTPUT_DIR}/host_removed/${base}_clean_1.fq.gz \
    ${OUTPUT_DIR}/host_removed/${base}_clean_2.fq.gz | \
bracken -d ${REFERENCE_DB}/bracken \
    -r 150 \
    -l 'S' \
    -t ${SLURM_CPUS_PER_TASK} \
    -o ${OUTPUT_DIR}/kraken2/${base}_bracken.txt
