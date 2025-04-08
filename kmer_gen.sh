#!/bin/bash

# Species-Specific K-mer Pipeline for Primer Design
# This script processes bacterial genomes to identify species-specific k-mers

# Configuration
KMER_SIZE=22
MIN_FREQ=1  # Minimum k-mer frequency to consider
THREADS=16  # Adjust based on available CPU cores

# Input directories
REFERENCE_GENOME="reference_genome.fasta"
INCLUSIVITY_DIR="inclusivity_genomes/"
EXCLUSIVITY_DIR="exclusivity_genomes/"

# Output directories
RESULTS_DIR="kmer_results/"
mkdir -p $RESULTS_DIR

echo "======= Species-Specific K-mer Pipeline ======="
echo "K-mer size: $KMER_SIZE"
echo "Reference genome: $REFERENCE_GENOME"
echo "Using $THREADS threads"

# Step 1: Generate k-mers from reference genome
echo "Generating ${KMER_SIZE}-mers from reference genome..."
jellyfish count -m $KMER_SIZE -s 100M -t $THREADS -C $REFERENCE_GENOME -o $RESULTS_DIR/reference_kmers.jf
jellyfish dump $RESULTS_DIR/reference_kmers.jf > $RESULTS_DIR/reference_kmers.fa

# Extract just the sequences without headers
grep -v ">" $RESULTS_DIR/reference_kmers.fa > $RESULTS_DIR/reference_kmers.txt
echo "Generated $(wc -l < $RESULTS_DIR/reference_kmers.txt) unique ${KMER_SIZE}-mers"

# Step 2: Process inclusivity genomes (target species)
echo "Processing inclusivity genomes..."
mkdir -p $RESULTS_DIR/inclusivity

# Combine all inclusivity genomes
cat $INCLUSIVITY_DIR/*.fasta > $RESULTS_DIR/inclusivity_combined.fasta

# Count k-mers in inclusivity genomes
jellyfish count -m $KMER_SIZE -s 100M -t $THREADS -C $RESULTS_DIR/inclusivity_combined.fasta -o $RESULTS_DIR/inclusivity_kmers.jf
jellyfish dump $RESULTS_DIR/inclusivity_kmers.jf > $RESULTS_DIR/inclusivity_kmers.fa

# Step 3: Process exclusivity genomes (non-target species)
echo "Processing exclusivity genomes..."
mkdir -p $RESULTS_DIR/exclusivity

# Combine all exclusivity genomes
cat $EXCLUSIVITY_DIR/*.fasta > $RESULTS_DIR/exclusivity_combined.fasta

# Count k-mers in exclusivity genomes
jellyfish count -m $KMER_SIZE -s 100M -t $THREADS -C $RESULTS_DIR/exclusivity_combined.fasta -o $RESULTS_DIR/exclusivity_kmers.jf
jellyfish dump $RESULTS_DIR/exclusivity_kmers.jf > $RESULTS_DIR/exclusivity_kmers.fa

# Step 4: Find species-specific k-mers
echo "Identifying species-specific k-mers..."

# Extract k-mer sequences
grep -v ">" $RESULTS_DIR/inclusivity_kmers.fa | sort > $RESULTS_DIR/inclusivity_sequences.txt
grep -v ">" $RESULTS_DIR/exclusivity_kmers.fa | sort > $RESULTS_DIR/exclusivity_sequences.txt

# Find k-mers present in reference AND inclusivity BUT NOT in exclusivity
comm -12 <(sort $RESULTS_DIR/reference_kmers.txt) <(sort $RESULTS_DIR/inclusivity_sequences.txt) > $RESULTS_DIR/ref_incl_common.txt
comm -23 $RESULTS_DIR/ref_incl_common.txt <(sort $RESULTS_DIR/exclusivity_sequences.txt) > $RESULTS_DIR/species_specific_kmers.txt

echo "Found $(wc -l < $RESULTS_DIR/species_specific_kmers.txt) species-specific ${KMER_SIZE}-mers"

# Step 5: Categorize k-mers
echo "Categorizing k-mers..."

# Calculate k-mer frequencies in inclusivity set
declare -A kmer_freq
while read -r kmer; do
    count=$(grep -c "^$kmer$" $RESULTS_DIR/inclusivity_sequences.txt)
    kmer_freq["$kmer"]=$count
done < $RESULTS_DIR/species_specific_kmers.txt

# Filter k-mers based on frequency and write to appropriate category
touch $RESULTS_DIR/kmers_PASS.txt $RESULTS_DIR/kmers_WARN.txt $RESULTS_DIR/kmers_FAIL.txt

# Calculate total inclusivity genomes
INCL_GENOME_COUNT=$(ls $INCLUSIVITY_DIR/*.fasta | wc -l)
echo "Total inclusivity genomes: $INCL_GENOME_COUNT"

# Categorize based on frequency threshold
PASS_THRESHOLD=$(echo "$INCL_GENOME_COUNT * 0.95" | bc | cut -d. -f1)
WARN_THRESHOLD=$(echo "$INCL_GENOME_COUNT * 0.75" | bc | cut -d. -f1)

echo "PASS threshold: >=$PASS_THRESHOLD genomes"
echo "WARN threshold: >=$WARN_THRESHOLD genomes"

for kmer in "${!kmer_freq[@]}"; do
    freq=${kmer_freq["$kmer"]}
    if [ "$freq" -ge "$PASS_THRESHOLD" ]; then
        echo "$kmer" >> $RESULTS_DIR/kmers_PASS.txt
    elif [ "$freq" -ge "$WARN_THRESHOLD" ]; then
        echo "$kmer" >> $RESULTS_DIR/kmers_WARN.txt
    else
        echo "$kmer" >> $RESULTS_DIR/kmers_FAIL.txt
    fi
done

echo "PASS k-mers: $(wc -l < $RESULTS_DIR/kmers_PASS.txt)"
echo "WARN k-mers: $(wc -l < $RESULTS_DIR/kmers_WARN.txt)"
echo "FAIL k-mers: $(wc -l < $RESULTS_DIR/kmers_FAIL.txt)"

# Step 6: Select the best candidates for primer design
echo "Selecting best candidate k-mers for primer design..."
# Filter for ideal GC content (40-60%)
cat $RESULTS_DIR/kmers_PASS.txt | while read kmer; do
    gc_count=$(echo $kmer | grep -o "[GC]" | wc -l)
    gc_percent=$(echo "scale=2; $gc_count * 100 / $KMER_SIZE" | bc)
    if (( $(echo "$gc_percent >= 40 && $gc_percent <= 60" | bc -l) )); then
        echo "$kmer"
    fi
done > $RESULTS_DIR/primer_candidates.txt

echo "Found $(wc -l < $RESULTS_DIR/primer_candidates.txt) suitable primer candidates"
echo "Results saved to $RESULTS_DIR/primer_candidates.txt"

echo "Pipeline completed successfully!"
