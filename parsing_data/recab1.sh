#!/bin/bash
set -euo pipefail

SAMPLE="sample"
LINKER_SEQ="GGTGGTGGTGGTTCTGGTGGTGGTGGTTCTGGTGGTGGTGGTTCT"
THREADS=8

RAW_DIR="raw"
REF_DIR="ref"
OUT_DIR="output_hybrid"
mkdir -p $OUT_DIR

# 1. Merge reads
bbmerge.sh in1=$RAW_DIR/${SAMPLE}_R1.fastq.gz \
           in2=$RAW_DIR/${SAMPLE}_R2.fastq.gz \
           out=$OUT_DIR/merged.fastq \
           outu1=$OUT_DIR/unmerged_R1.fastq \
           outu2=$OUT_DIR/unmerged_R2.fastq \
           threads=$THREADS

cat $OUT_DIR/merged.fastq $OUT_DIR/unmerged_R1.fastq $OUT_DIR/unmerged_R2.fastq > $OUT_DIR/all_reads.fastq

# 2. IgBLAST
igblastn -germline_db_V $REF_DIR/sheep_gl_V.fasta \
         -germline_db_D $REF_DIR/sheep_gl_D.fasta \
         -germline_db_J $REF_DIR/sheep_gl_J.fasta \
         -organism sheep \
         -domain_system imgt \
         -ig_seqtype Ig \
         -num_threads $THREADS \
         -query $OUT_DIR/all_reads.fastq \
         -out $OUT_DIR/igblast.tsv \
         -outfmt 19

# 3. Extract high-confidence "seed" scFvs (linker-spanning)
python3 scripts/extract_scFv_by_linker.py \
    --fasta $OUT_DIR/all_reads.fastq \
    --igblast_tsv $OUT_DIR/igblast.tsv \
    --linker "$LINKER_SEQ" \
    --max_mismatch 1 \
    --output_fasta $OUT_DIR/seeds.fasta \
    --output_tsv $OUT_DIR/seeds.tsv

# 4. Map all reads to seeds to extend/validate
minimap2 -x map-ont -d $OUT_DIR/seeds.mmi $OUT_DIR/seeds.fasta
minimap2 -ax map-ont $OUT_DIR/seeds.mmi $OUT_DIR/all_reads.fastq | samtools sort -@ $THREADS -o $OUT_DIR/aligned.bam
samtools index $OUT_DIR/aligned.bam

# 5. Generate consensus per seed (optional: use racon or custom script)
# For simplicity, we keep seeds as final output
cp $OUT_DIR/seeds.fasta $OUT_DIR/scFv_full_length.fasta
cp $OUT_DIR/seeds.tsv $OUT_DIR/scFv_metadata.tsv

echo "Hybrid assembly done. Seeds used as high-confidence scFvs."
