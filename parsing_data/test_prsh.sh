#!/usr/bin/env bash

# === Input ===
# Adjust these paths
INPUT_DIR="/path/to/demultiplexed_fastqs"
OUTPUT_DIR="/path/to/output"
IGBLAST_DB="/path/to/igblast_db"   # pre-downloaded IgBlast germline DB
NUM_THREADS=8

mkdir -p "${OUTPUT_DIR}/presto" "${OUTPUT_DIR}/igblast"

# Loop through each sample
for R1 in ${INPUT_DIR}/*_R1.fastq.gz; do
    SAMPLE=$(basename ${R1} _R1.fastq.gz)
    R2=${INPUT_DIR}/${SAMPLE}_R2.fastq.gz
    
    echo "=== Processing $SAMPLE ==="

    # 1. Preprocessing with pRESTO
    # Step 1a: Quality filter
    FilterSeq.py quality \
        -s ${R1} \
        -q 20 \
        --nproc ${NUM_THREADS} \
        --outdir ${OUTPUT_DIR}/presto \
        --outname ${SAMPLE}_R1.filtered

    FilterSeq.py quality \
        -s ${R2} \
        -q 20 \
        --nproc ${NUM_THREADS} \
        --outdir ${OUTPUT_DIR}/presto \
        --outname ${SAMPLE}_R2.filtered

    # Step 1b: Pair reads
    PairSeq.py \
        -1 ${OUTPUT_DIR}/presto/${SAMPLE}_R1.filtered.fastq \
        -2 ${OUTPUT_DIR}/presto/${SAMPLE}_R2.filtered.fastq \
        --outdir ${OUTPUT_DIR}/presto \
        --outname ${SAMPLE}.paired

    # 2. Run IgBlast
    # (convert to FASTA first)
    SeqIO.py fastq2fasta \
        -s ${OUTPUT_DIR}/presto/${SAMPLE}.paired.fastq \
        --outdir ${OUTPUT_DIR}/igblast \
        --outname ${SAMPLE}.fasta

    igblastn \
        -query ${OUTPUT_DIR}/igblast/${SAMPLE}.fasta \
        -germline_db_V ${IGBLAST_DB}/human_gl_V \
        -germline_db_D ${IGBLAST_DB}/human_gl_D \
        -germline_db_J ${IGBLAST_DB}/human_gl_J \
        -organism human \
        -domain_system imgt \
        -num_threads ${NUM_THREADS} \
        -outfmt 7 \
        -out ${OUTPUT_DIR}/igblast/${SAMPLE}.igblast.out

    echo "=== Done $SAMPLE ==="
done
