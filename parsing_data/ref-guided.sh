fastp -i R1.fastq.gz -I R2.fastq.gz -o R1.clean.fastq.gz -O R2.clean.fastq.gz --thread 8
pandaseq -f R1.clean.fastq.gz -r R2.clean.fastq.gz -w merged.fastq
# map permissively with bwa mem (lower mismatch penalty, or use minimap2 asm5)
bwa mem -B 2 -O 6 -E 1 scfv_ref.fasta R1.clean.fastq.gz R2.clean.fastq.gz | samtools view -b - | samtools sort -o mapped.bam
samtools index mapped.bam

# extract read names that align (primary or secondary)
samtools view mapped.bam | awk '{print $1}' | sort | uniq > mapped_readnames.txt

# extract reads from original fastq (paired)
seqtk subseq R1.clean.fastq.gz mapped_readnames.txt > R1.mapped.fastq
seqtk subseq R2.clean.fastq.gz mapped_readnames.txt > R2.mapped.fastq

# if you used merged.fastq and want those:
seqtk subseq merged.fastq mapped_readnames.txt > merged.mapped.fastq || true

# convert mapped reads/merged to fasta if needed
seqtk seq -a merged.mapped.fastq > merged.mapped.fasta

# run igblastn (assume sheep IMGT DB)
igblastn -query merged.mapped.fasta \
         -germline_db_V ../imgt/sheep_V.fasta \
         -germline_db_D ../imgt/sheep_D.fasta \
         -germline_db_J ../imgt/sheep_J.fasta \
         -organism sheep \
         -outfmt "6 qseqid qseq sseqid qstart qend sstart send evalue bitscore" \
         -out igblast_mapped.tsv

# assume you have a fasta file `HCDR3_upstream.fa` (one sequence per read)
vsearch --cluster_fast HCDR3_upstream.fa --id 0.90 --centroids clonotypes_centroids.fa --uc clonotypes.uc --threads 8

# extract member read names using clonotypes.uc
# then subset R1/R2 mapped fastqs (seqtk subseq)
# map cluster reads to reference:
bwa mem -t 8 scfv_ref.fasta cluster_R1.fastq cluster_R2.fastq | samtools sort -o cluster.bam
samtools index cluster.bam
# call consensus as in A:
samtools mpileup -f scfv_ref.fasta cluster.bam | bcftools call -mv -Oz -o cluster.vcf.gz
bcftools index cluster.vcf.gz
bcftools consensus -f scfv_ref.fasta cluster.vcf.gz > cluster_consensus.fa


spades.py --only-assembler -1 cluster_R1.fastq -2 cluster_R2.fastq -s cluster_merged.fastq -o spades_cluster -t 8 -k 21,33,55,77,99,127 --careful
# then search spades_cluster/contigs.fasta for contig ~800bp and annotate with igblast

