fastp -i R1.fastq.gz -I R2.fastq.gz -o R1.clean.fastq.gz -O R2.clean.fastq.gz \
      --qualified_quality_phred 20 --length_required 50 --thread 8 --html fastp.html

      pandaseq -f R1.clean.fastq.gz -r R2.clean.fastq.gz -w merged.fastq
# or BBMerge:
# bbmerge.sh in1=R1.clean.fastq.gz in2=R2.clean.fastq.gz out=merged.fastq

# index
bwa index scfv_ref.fasta

# map (paired end mapping)
bwa mem -t 8 scfv_ref.fasta R1.clean.fastq.gz R2.clean.fastq.gz | samtools view -b - | samtools sort -o mapped.bam
samtools index mapped.bam

samtools flagstat mapped.bam > flagstat.txt
samtools depth -a mapped.bam | awk '{sum+=$3; cnt++} END{print "mean_cov", sum/cnt; print "positions", cnt}' > cov_summary.txt

# generate mpileup
samtools mpileup -f scfv_ref.fasta mapped.bam -Q 20 -A -d 2000000 | bcftools call -mv -Oz -o calls.vcf.gz
bcftools index calls.vcf.gz

# optional filter
bcftools filter -s LOWQUAL -e 'QUAL<20 || DP<10' calls.vcf.gz -Oz -o calls.filtered.vcf.gz
bcftools index calls.filtered.vcf.gz

# make consensus sequence (apply filtered vcf)
bcftools consensus -f scfv_ref.fasta calls.filtered.vcf.gz > scfv_consensus.fa

# translate (simple Biopython / transeq)
transeq -sequence scfv_consensus.fa -outseq scfv_consensus.aa.fasta
# or python quick check for stop codons

----------

# Map cluster reads to reference
bwa mem -t 8 scfv_ref.fasta cluster_R1.fastq cluster_R2.fastq | samtools sort -o cluster.bam
samtools index cluster.bam

# Generate pileup + call variants
bcftools mpileup -Ou -f scfv_ref.fasta cluster.bam | \
  bcftools call -mv -Oz -o cluster.vcf.gz

# Index VCF
bcftools index cluster.vcf.gz

# Create consensus FASTA
bcftools consensus -f scfv_ref.fasta cluster.vcf.gz > cluster_consensus.fa
