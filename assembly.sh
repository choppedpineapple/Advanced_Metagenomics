#!/bin/bash
set -euo pipefail

# Validate inputs
if [ $# -ne 2 ]; then
    echo "Usage: $0 <R1.fastq.gz> <R2.fastq.gz>"
    exit 1
fi

# Initialize variables
r1=$1
r2=$2
k=21  # Optimal k-mer size for bacterial genomes[3][7]
min_occurrence=3  # Error correction threshold

# Preprocessing
preprocess_reads() {
    zcat "$1" | \
    awk 'NR%4==2 {print}' | \
    tr '[:lower:]' '[:upper:]' | \
    awk -v k=$k 'length($0)>=k {print substr($0,1,k)}'
}

# Build k-mer hash
declare -A kmers
while read -r kmer; do
    ((kmers[$kmer]++))
done < <(preprocess_reads "$r1"; preprocess_reads "$r2")

# Contig construction
current_contig=""
for kmer in "${!kmers[@]}"; do
    if [ ${kmers[$kmer]} -ge $min_occurrence ]; then
        if [ -z "$current_contig" ]; then
            current_contig=$kmer
        else
            overlap=$((k-1))
            if [[ "${current_contig: -$overlap}" == "${kmer:0:$overlap}" ]]; then
                current_contig+="${kmer:$overlap}"
            fi
        fi
    fi
done

# Output assembly
echo ">contig_1 length_${#current_contig}"
fold -w 80 <<< "$current_contig"
