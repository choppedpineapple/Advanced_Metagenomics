#!/bin/bash

# Configuration
THREADS=16
KRAKEN2_DB="/path/to/microbial_db" # Exclude plants (e.g., Standard-2024)
KNEADDATA_DB="/path/to/host_db" # Bowtie2 human/plant index
HUMANN3_DB="/path/to/uniref90_metacyc"
SAMPLE_PREFIX="sample1_S1"

# 1. Trimming & De-hosting with kneaddata
kneaddata --input ${SAMPLE_PREFIX}_L001_R1_001.fastq.gz \
          --input ${SAMPLE_PREFIX}_L001_R2_001.fastq.gz \
          --output kneaddata_out \
          --reference-db $KNEADDATA_DB \
          --trimmomatic-options "SLIDINGWINDOW:4:20 MINLEN:50" \
          --bowtie2-options "--very-sensitive --threads $THREADS" \
          --remove-intermediate-output \
          --threads $THREADS

# 2. Kraken2 Classification
kraken2 --db $KRAKEN2_DB \
        --threads $THREADS \
        --paired kneaddata_out/*_paired_1.fastq kneaddata_out/*_paired_2.fastq \
        --output kraken2_output.txt \
        --report kraken2_report.txt

# 3. Filter Plant Reads (Viridiplantae: 33090)
kraken2-filter --db $KRAKEN2_DB \
               --exclude 33090 \
               --input kraken2_output.txt \
               --output kraken2_filtered.txt

# 4. Bracken Abundance Estimation
bracken -d $KRAKEN2_DB \
        -i kraken2_report.txt \
        -o bracken_output.tsv \
        -r 150 \
        -t $THREADS

# 5. Convert Bracken to HUMAnN3-compatible MetaPhlAn format
awk 'BEGIN{OFS="\t"}{print $1,$6}' bracken_output.tsv | grep 's__' > metaphlan_output.tsv

# 6. HUMAnN3 Functional Analysis
humann --input kneaddata_out/*_paired_1.fastq \
       --output humann3_out \
       --threads $THREADS \
       --taxonomic-profile metaphlan_output.tsv \
       --protein-database $HUMANN3_DB \
       --bypass-nucleotide-search \
       --bypass-translated-search

# 7. Alpha Diversity with KrakenTools
kreport2mpa.py -r kraken2_report.txt -o mpa_report.txt
python3 krakentools/alpha_diversity.py --input mpa_report.txt --metric shannon

# 8. Visualization
humann_barplot --input humann3_out/*_pathabundance.tsv --output pathway_plot.html
ktImportTaxonomy -q 2 -t 5 kraken2_filtered.txt -o krona_plot.html
