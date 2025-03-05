#!/bin/bash
THREADS=16
KRAKEN_DB="/path/to/kraken2_db" # Include human+plant in DB for proper classification
HUMAN_REF="/path/to/human_genome.fa"
FASTP_OPTS="--detect_adapter_for_pe --trim_poly_g --correction --cut_right --cut_window_size 4 --cut_mean_quality 20"

# Step 1: Quality control with fastp
for R1 in *R1_001.fastq.gz; do
    R2=${R1/R1_001/R2_001}
    fastp -i $R1 -I $R2 \
        -o ${R1%.fastq.gz}_trimmed.fastq.gz \
        -O ${R2%.fastq.gz}_trimmed.fastq.gz \
        $FASTP_OPTS \
        -w $THREADS \
        -j ${R1%_R1_001.fastq.gz}.json \
        -h ${R1%_R1_001.fastq.gz}.html
done

# Step 2: Host read removal using Bowtie2 (more sensitive than Kraken-based removal)
bowtie2-build $HUMAN_REF human_ref > /dev/null
for R1 in *trimmed.fastq.gz; do
    R2=${R1/R1/R2}
    bowtie2 -x human_ref \
        -1 $R1 -2 $R2 \
        --un-conc-gz ${R1%_R1_001_trimmed.fastq.gz}_dehosted%.fq.gz \
        --threads $THREADS \
        --very-sensitive \
        --dovetail \
        --no-unal \
        > /dev/null 2> ${R1%_R1_001_trimmed.fastq.gz}_bowtie2.log
done

# Step 3: Kraken2 classification with full database
for R1 in *dehosted_1.fq.gz; do
    R2=${R1/_1.fq.gz/_2.fq.gz}
    kraken2 --db $KRAKEN_DB \
        --paired $R1 $R2 \
        --threads $THREADS \
        --report ${R1%_1.fq.gz}.kreport \
        --output ${R1%_1.fq.gz}.kraken2
done

# Step 4: Bracken abundance estimation
for KREPORT in *.kreport; do
    bracken -d $KRAKEN_DB \
        -i $KREPORT \
        -o ${KREPORT%.kreport}.bracken \
        -l S \
        -r 150 \
        -t 10
done

# Step 5: Post-processing with KrakenTools
combine_bracken_outputs.py --inputs *.bracken -o combined_abundance.tsv
filter_table.py -t combined_abundance.tsv \
    --exclude-taxa 9606,33090 \ # Human and Viridiplantae
    -o filtered_abundance.tsv

# Step 6: Functional analysis (parallel track)
for R1 in *dehosted_1.fq.gz; do
    R2=${R1/_1.fq.gz/_2.fq.gz}
    metaphlan $R1,$R2 \
        --input_type fastq \
        --nproc $THREADS \
        --bowtie2out ${R1%_1.fq.gz}.bt2out \
        -o ${R1%_1.fq.gz}_metaphlan4.tsv
done

merge_metaphlan_tables.py *_metaphlan4.tsv > merged_metaphlan4.tsv

for R1 in *dehosted_1.fq.gz; do
    humann --input $R1 \
        --output humann3_out \
        --threads $THREADS \
        --protein-database uniref90 \
        --metaphlan-options "-t rel_ab" \
        --resume
done

# Step 7: Diversity analysis (requires R with phyloseq)
echo 'library(phyloseq); library(tidyverse);
ab_table <- read_tsv("filtered_abundance.tsv") %>% 
  column_to_rownames("taxon_id") %>%
  otu_table(taxa_are_rows = TRUE)
alpha_div <- estimate_richness(ab_table, measures=c("Shannon"))
write_tsv(alpha_div, "alpha_diversity.tsv")' > diversity.R

Rscript diversity.R
