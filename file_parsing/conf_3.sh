# Extract the first 25 bases of the reads to verify primer sequence presence and variance
seqkit subseq -r 1:25 sample-WPDL.fastq | seqkit seq -s | sort | uniq -c | sort -nr | head -n 20
