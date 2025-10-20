# if you have paired FASTQ per cluster, run:
spades.py -1 cluster_X_R1.fastq -2 cluster_X_R2.fastq -s cluster_X_merged.fastq -o cluster_X_spades -k 21,33,55,77,99,127 --careful -t 8

# if you only have single merged FASTA/FASTQ:
spades.py -s cluster_X.fasta -o cluster_X_spades -k 21,33,55,77,99,127 --careful -t 8
