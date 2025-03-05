#!/bin/bash

# Configuration
THREADS=16
KRAKEN2_DB="/path/to/kraken2_db"
BRACKEN_DB="/path/to/bracken_db"
HOST_DB="/path/to/host_db" # Combined human + plant Bowtie2 index
HUMANN3_DB="/path/to/humann3_db"
SAMPLE_PREFIX="sample1_S1"

# 1. Trimming & QC with fastp
fastp -i ${SAMPLE_PREFIX}_L001_R1_001.fastq.gz \
      -I ${SAMPLE_PREFIX}_L001_R2_001.fastq.gz \
      -o trimmed_1.fq.gz \
      -O trimmed_2.fq.gz \
      --trim_poly_g --correction \
      --thread $THREADS \
      --html fastp_report.html \
      --json fastp_report.json

# 2. De-hosting with Bowtie2 (remove human/plant reads)
bowtie2 -x $HOST_DB \
         -1 trimmed_1.fq.gz -2 trimmed_2.fq.gz \
         --un-conc-gz dehosted_%.fq.gz \
         --threads $THREADS \
         -S host_mapped.sam > bowtie2_log.txt 2>&1

# 3. Kraken2 taxonomic classification
kraken2 --db $KRAKEN2_DB \
        --threads $THREADS \
        --paired dehosted_1.fq.gz dehosted_2.fq.gz \
        --output kraken2_output.txt \
        --report kraken2_report.txt

# 4. Bracken abundance estimation
bracken -d $BRACKEN_DB \
        -i kraken2_report.txt \
        -o bracken_output.tsv \
        -r 150 \
        -t $THREADS

# 5. Generate reports with krakentools
kreport2mpa.py -r kraken2_report.txt -o mpa_report.txt
# For visualization, consider Krona: ktImportTaxonomy -q 2 -t 5 kraken2_output.txt -o krona_report.html

# 6. HUMAnN3 functional analysis (skip MetaPhlAn to avoid redundancy)
humann --input dehosted_1.fq.gz \
       --output humann3_output \
       --threads $THREADS \
       --bypass-nucleotide-search \
       --bypass-translated-search \
       --taxonomic-profile bracken_output.tsv # Use Bracken's profile

# 7. Alpha Diversity (Shannon Index from Bracken output)
# Install pandas if needed: pip install pandas
python3 - <<EOF
import pandas as pd
from scipy.stats import entropy

df = pd.read_csv('bracken_output.tsv', sep='\t', comment='#')
species = df[df['taxonomy_lvl'] == 'S']
abundance = species['fraction_total_reads']
shannon = entropy(abundance, base=2)
print(f"Shannon Index: {shannon}")
EOF
