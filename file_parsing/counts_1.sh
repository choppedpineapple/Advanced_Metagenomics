vsearch \
    --threads 32 \
    --search_pcr linker.fa \
    --fastq scfv_reads.fastq \
    --maxdiffs 6 \
    --strand both \
    --minamp 41 \
    --maxamp 41 \
    --qmask none \
    --output_no_hits \
    --alnout matches.aln

  grep "^Sbjct" matches.aln | awk '{print $3}' | sort | uniq -c | sort -nr > linker_counts.txt

##########

ugrep -I -Z6 -o -N -j32 "ACGT...41bp..." scfv_reads.fastq > raw_hits.txt

grep -o '[ACGTN]\{41\}' raw_hits.txt | sort | uniq -c | sort -nr > linker_counts.txt


###########

fuzznuc -sequence scfv_reads.fastq \
        -pattern linker.fa \
        -pmismatch 6 \
        -rformat excel \
        -outfile fuzznuc_output.txt
