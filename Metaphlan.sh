#!/bin/bash

# Script to run MetaPhlAn 4 on paired-end Illumina data, detecting all taxa.
#
# Usage: ./run_metaphlan4.sh <forward_reads.fastq.gz> <reverse_reads.fastq.gz> <output_basename> <num_threads>
#
# Example: ./run_metaphlan4.sh sample1_R1.fastq.gz sample1_R2.fastq.gz sample1_output 4

# --- Input Parameters ---
forward_reads="$1"
reverse_reads="$2"
output_basename="$3"
num_threads="$4"

# --- Check Input ---

if [ -z "$forward_reads" ] || [ -z "$reverse_reads" ] || [ -z "$output_basename" ] || [ -z "$num_threads" ]; then
  echo "Error:  Missing input parameters."
  echo "Usage:  $0 <forward_reads.fastq.gz> <reverse_reads.fastq.gz> <output_basename> <num_threads>"
  exit 1
fi

if [ ! -f "$forward_reads" ] || [ ! -f "$reverse_reads" ]; then
    echo "Error: Input FASTQ files not found."
    exit 1
fi

# --- MetaPhlAn 4 Command ---

metaphlan "$forward_reads","$reverse_reads" \
  --input_type fastq \
  --nproc "$num_threads" \
  --bowtie2out "${output_basename}.bowtie2.bz2" \
  --output_file "${output_basename}.txt" \
  --unclassified_estimation \
  --tax_level s  \
  --add_viruses \
  --stat_q 0.0 # Very accurate. Sets the percentile of average read length. Lowering the value make the analysis more accurate (and slower)

# Explanation of Options:
#
# * "$forward_reads","$reverse_reads":  Provides comma-separated paired-end reads.
# * --input_type fastq:              Specifies input file type.
# * --nproc "$num_threads":             Sets the number of threads for Bowtie2 (alignment) and MetaPhlAn.
# * --bowtie2out:                    Saves the Bowtie2 alignment output (SAM format, compressed).  This is useful for debugging or downstream analysis.
# * --output_file:                   Specifies the main output file name (tab-separated text).
# * --unclassified_estimation:      Include reads that map to multiple species in the estimation.  This is crucial for accurate abundance estimates.
# * --tax_level s:                  Reports abundances at the species level.  You can change this to 'g' (genus), 'f' (family), etc.
# * --add_viruses:                   Include virus detection.  This is *essential* for finding viruses.
# * --stat_q 0.0:                     Sets a stricter threshold for read inclusion, increasing accuracy. Lowering the value makes it more strict.

# --- Additional Processing (Optional but Recommended) ---

# 1.  Create a BIOM file (for compatibility with other tools)
biom convert -i "${output_basename}.txt" -o "${output_basename}.biom" --table-type="OTU table" --to-hdf5

# 2.  Generate a Bowtie2 index of the database (optional, but speeds up subsequent runs on the *same* database, which may not be your situation):
#     This should ONLY be done once for a given database version. It is best done separately.

# 3.  Convert SAM to BAM and sort (optional, requires samtools):  If you want a sorted BAM file for visualization or other analysis.
if command -v samtools &> /dev/null; then
  echo "Converting Bowtie2 output to sorted BAM..."
  samtools view -@ "$num_threads" -S -b "${output_basename}.bowtie2.bz2" | samtools sort -@ "$num_threads" -o "${output_basename}.sorted.bam"
  samtools index "${output_basename}.sorted.bam"
  echo "BAM file created: ${output_basename}.sorted.bam"
else
  echo "samtools not found. Skipping BAM conversion."
fi



echo "MetaPhlAn 4 analysis complete."
echo "Results are in: ${output_basename}.txt"
echo "Bowtie2 output (compressed SAM): ${output_basename}.bowtie2.bz2"
echo "BIOM file: ${output_basename}.biom"


exit 0
