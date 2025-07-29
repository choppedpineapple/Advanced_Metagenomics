#!/bin/bash

# Host read removal from paired-end metagenomic data using BWA MEM
# Usage: ./remove_host_reads.sh <R1.fastq> <R2.fastq> <output_prefix> <human_ref.fasta>

# Check if required arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <R1.fastq> <R2.fastq> <output_prefix> <human_ref.fasta>"
    echo "Example: $0 sample_R1.fastq sample_R2.fastq clean_sample GRCh38_latest_genomic.fna"
    exit 1
fi

# Input parameters
R1_INPUT=$1
R2_INPUT=$2
OUTPUT_PREFIX=$3
HUMAN_REF=$4

# Check if input files exist
if [ ! -f "$R1_INPUT" ] || [ ! -f "$R2_INPUT" ] || [ ! -f "$HUMAN_REF" ]; then
    echo "Error: One or more input files do not exist"
    exit 1
fi

# Set number of threads (adjust based on your system)
THREADS=8

echo "Starting host read removal pipeline..."
echo "R1 input: $R1_INPUT"
echo "R2 input: $R2_INPUT"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Human reference: $HUMAN_REF"
echo "Threads: $THREADS"

# Step 1: Index the human reference genome (if not already indexed)
echo "Checking BWA index..."
if [ ! -f "${HUMAN_REF}.bwt" ]; then
    echo "Creating BWA index for human reference genome..."
    bwa index "$HUMAN_REF"
else
    echo "BWA index already exists"
fi

# Step 2: Align reads to human reference using BWA MEM
echo "Aligning reads to human reference..."
SAM_FILE="${OUTPUT_PREFIX}_aligned.sam"
bwa mem -t "$THREADS" "$HUMAN_REF" "$R1_INPUT" "$R2_INPUT" > "$SAM_FILE"

# Step 3: Convert SAM to BAM and sort
echo "Converting SAM to BAM and sorting..."
BAM_FILE="${OUTPUT_PREFIX}_aligned_sorted.bam"
samtools view -bS "$SAM_FILE" | samtools sort -@ "$THREADS" -o "$BAM_FILE"

# Step 4: Extract unmapped reads (non-human reads)
echo "Extracting unmapped reads..."
UNMAPPED_BAM="${OUTPUT_PREFIX}_unmapped.bam"

# Extract reads where both reads in pair are unmapped (flag 12 = 4+8)
# -f 12: include reads with flags 4 (unmapped) and 8 (mate unmapped)
# -F 256: exclude secondary alignments
samtools view -b -f 12 -F 256 "$BAM_FILE" > "$UNMAPPED_BAM"

# Step 5: Convert unmapped BAM back to FASTQ files
echo "Converting unmapped reads back to FASTQ..."
OUTPUT_R1="${OUTPUT_PREFIX}_clean_R1.fastq"
OUTPUT_R2="${OUTPUT_PREFIX}_clean_R2.fastq"

# Sort by read name for proper pairing
samtools sort -n -@ "$THREADS" "$UNMAPPED_BAM" | \
samtools fastq -1 "$OUTPUT_R1" -2 "$OUTPUT_R2" -0 /dev/null -s /dev/null -n

# Step 6: Generate statistics
echo "Generating statistics..."
STATS_FILE="${OUTPUT_PREFIX}_removal_stats.txt"

TOTAL_READS_R1=$(wc -l < "$R1_INPUT" | awk '{print $1/4}')
CLEAN_READS_R1=$(wc -l < "$OUTPUT_R1" | awk '{print $1/4}')
REMOVED_READS=$((TOTAL_READS_R1 - CLEAN_READS_R1))
REMOVAL_PERCENTAGE=$(awk "BEGIN {printf \"%.2f\", ($REMOVED_READS/$TOTAL_READS_R1)*100}")

cat > "$STATS_FILE" << EOF
Host Read Removal Statistics
============================
Original read pairs: $TOTAL_READS_R1
Clean read pairs: $CLEAN_READS_R1
Removed read pairs: $REMOVED_READS
Removal percentage: $REMOVAL_PERCENTAGE%

Input files:
- R1: $R1_INPUT
- R2: $R2_INPUT

Output files:
- Clean R1: $OUTPUT_R1
- Clean R2: $OUTPUT_R2

Reference genome: $HUMAN_REF
EOF

echo "Statistics written to: $STATS_FILE"

# Step 7: Clean up intermediate files (optional)
echo "Cleaning up intermediate files..."
rm -f "$SAM_FILE" "$BAM_FILE" "$UNMAPPED_BAM"

# Step 8: Compress output FASTQ files (optional)
echo "Compressing output files..."
gzip "$OUTPUT_R1" "$OUTPUT_R2"

echo "Host read removal completed!"
echo "Clean reads saved as:"
echo "- ${OUTPUT_R1}.gz"
echo "- ${OUTPUT_R2}.gz"
echo "Check $STATS_FILE for detailed statistics" 
