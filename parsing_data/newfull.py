#!/bin/bash

# Input files
REF="transferrin_scfv_ref.fasta"   # 738 bp reference
R1="sample_R1.fastq.gz"
R2="sample_R2.fastq.gz"
OUTPREFIX="transferrin_scfv"

# 1. Build Bowtie2 index
bowtie2-build $REF ${OUTPREFIX}_index

# 2. Align reads
bowtie2 -x ${OUTPREFIX}_index -1 $R1 -2 $R2 -S ${OUTPREFIX}.sam --very-sensitive-local -p 8

# 3. Convert SAM to sorted BAM
samtools view -bS ${OUTPREFIX}.sam | samtools sort -o ${OUTPREFIX}_sorted.bam

# 4. Index BAM
samtools index ${OUTPREFIX}_sorted.bam

# 5. Generate pileup
samtools mpileup -aa -A -d 0 -B -Q 0 ${OUTPREFIX}_sorted.bam > ${OUTPREFIX}.mpileup

# 6. Generate consensus FASTA
cat ${OUTPREFIX}.mpileup | \
  awk '{if($4==0) next; print ">"NR"\n"$3}' | \
  awk 'BEGIN{RS=">";ORS=""}NR>1{print ">"$0}' > ${OUTPREFIX}_consensus.fasta

# OR, if you have bcftools installed (preferred for accuracy)
# bcftools mpileup -Ou -f $REF ${OUTPREFIX}_sorted.bam | bcftools call -mv -Oz -o ${OUTPREFIX}.vcf.gz
# bcftools index ${OUTPREFIX}.vcf.gz
# cat $REF | bcftools consensus ${OUTPREFIX}.vcf.gz > ${OUTPREFIX}_consensus.fasta
