# Filter R1 only
presto FilterSeq \
  -1 Transferrin-heavy_S2_L001_R1_001.fastq.gz \
  --minqual 30 --maxns 0 --minlen 250 \
  --outname Transferrin-heavy-R1

presto SeqConv --fastq Transferrin-heavy-R1.fastq --fasta Transferrin-heavy-R1.fasta

# Run IgBLAST on R1
igblastn -germline_db_V sheep_IGHV.fasta ... -query Transferrin-heavy-R1.fasta ...
