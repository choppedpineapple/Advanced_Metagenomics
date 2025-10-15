seqkit grep -s -i -r -p GGTGGAGGCGGTTCAGGCGGAGGTGGAGG merged.fastq > linker_reads.fastq

seqkit locate -p GGTGGAGGCGGTTCAGGCGGAGGTGGAGG linker_reads.fastq \
| awk '{start=$2-90; if(start<1) start=1; print $1"\t"start"\t"$2}' > upstream_regions.bed

seqkit subseq -f upstream_regions.bed linker_reads.fastq > upstream_90bp.fa

vsearch --cluster_fast upstream_90bp.fa --id 0.9 --centroids clonotypes.fa


spades.py -s cluster1_reads.fastq -k 127 -o cluster1_assembly

-----------------

# 1. Convert fastq → fasta if not already
seqtk seq -a merged.fastq > merged.fasta

# 2. Find linker matches and save coordinates
seqkit locate -p "GGTGGTGGTGGTAGC" -i merged.fasta > locate.tsv

# 3. Extract 90 bases upstream of each linker occurrence
seqkit subseq -f locate.tsv -u 90 merged.fasta > upstream_90.fa
