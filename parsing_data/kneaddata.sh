#!/bin/bash

# Host read removal using KneadData (recommended for metagenomics)
# KneadData combines quality filtering and host removal in one step
# Usage: ./kneaddata_dehost.sh <R1.fastq> <R2.fastq> <output_dir> <db_path>

# Check if required arguments are provided
if [ $# -ne 4 ]; then
    echo "Usage: $0 <R1.fastq> <R2.fastq> <output_dir> <human_db_path>"
    echo "Example: $0 sample_R1.fastq sample_R2.fastq ./clean_output /path/to/human_bowtie2_db"
    exit 1
fi

# Input parameters
R1_INPUT=$1
R2_INPUT=$2
OUTPUT_DIR=$3
HUMAN_DB=$4

# Check if input files exist
if [ ! -f "$R1_INPUT" ] || [ ! -f "$R2_INPUT" ]; then
    echo "Error: Input FASTQ files do not exist"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Set number of threads
THREADS=8

# Extract sample name from R1 filename
SAMPLE_NAME=$(basename "$R1_INPUT" | sed 's/_R1.*$//' | sed 's/\.fastq.*$//')

echo "Starting KneadData host removal pipeline..."
echo "Sample: $SAMPLE_NAME"
echo "R1 input: $R1_INPUT"
echo "R2 input: $R2_INPUT"
echo "Output directory: $OUTPUT_DIR"
echo "Human database: $HUMAN_DB"
echo "Threads: $THREADS"

# Run KneadData
echo "Running KneadData..."
kneaddata \
    --input "$R1_INPUT" \
    --input "$R2_INPUT" \
    --reference-db "$HUMAN_DB" \
    --output "$OUTPUT_DIR" \
    --output-prefix "$SAMPLE_NAME" \
    --threads "$THREADS" \
    --trimmomatic-options "SLIDINGWINDOW:4:20 MINLEN:50" \
    --bowtie2-options "--very-sensitive --dovetail" \
    --remove-intermediate-output

echo "KneadData completed!"
echo "Check ${OUTPUT_DIR}/${SAMPLE_NAME}.log for detailed statistics"
echo "Clean paired reads: ${OUTPUT_DIR}/${SAMPLE_NAME}_paired_1.fastq ${OUTPUT_DIR}/${SAMPLE_NAME}_paired_2.fastq"

# Alternative faster Bowtie2-only approach (if you don't want quality trimming)
echo ""
echo "=== Alternative: Bowtie2-only approach (faster, no quality trimming) ==="

#!/bin/bash
# Bowtie2-only host removal (faster alternative)
# Usage: ./bowtie2_dehost.sh <R1.fastq> <R2.fastq> <output_prefix> <bowtie2_index>

bowtie2_dehost() {
    local R1=$1
    local R2=$2
    local PREFIX=$3
    local BT2_INDEX=$4
    local THREADS=8
    
    echo "Running Bowtie2 host removal..."
    
    # Align to human genome and keep only unmapped reads
    bowtie2 \
        -x "$BT2_INDEX" \
        -1 "$R1" \
        -2 "$R2" \
        --threads "$THREADS" \
        --very-sensitive \
        --un-conc-gz "${PREFIX}_clean_R%.fastq.gz" \
        --al-conc-gz "${PREFIX}_human_R%.fastq.gz" \
        -S /dev/null
    
    echo "Bowtie2 host removal completed!"
    echo "Clean reads: ${PREFIX}_clean_R1.fastq.gz ${PREFIX}_clean_R2.fastq.gz"
    echo "Human reads: ${PREFIX}_human_R1.fastq.gz ${PREFIX}_human_R2.fastq.gz"
}

# Uncomment to run Bowtie2-only approach:
# bowtie2_dehost "$R1_INPUT" "$R2_INPUT" "${OUTPUT_DIR}/${SAMPLE_NAME}" "$HUMAN_DB"

# Database setup instructions
cat << 'EOF'

=== Database Setup Instructions ===

1. For KneadData with Bowtie2 (recommended):
   # Download and build human reference
   wget ftp://ftp.ncbi.nlm.nih.gov/genomes/all/GCF/000/001/405/GCF_000001405.39_GRCh38.p13/GCF_000001405.39_GRCh38.p13_genomic.fna.gz
   gunzip GCF_000001405.39_GRCh38.p13_genomic.fna.gz
   
   # Build Bowtie2 index
   bowtie2-build GCF_000001405.39_GRCh38.p13_genomic.fna human_bowtie2_db
   
   # Use the prefix 'human_bowtie2_db' as your database path

2. Pre-built databases (faster setup):
   # KneadData can download pre-built databases
   kneaddata_database --download human_genome bowtie2 /path/to/db/directory

3. Installation:
   # Install via conda (recommended)
   conda install -c bioconda kneaddata
   
   # Or via pip
   pip install kneaddata

=== Tool Comparison ===

KneadData advantages:
- Designed specifically for metagenomics
- Combines quality trimming + host removal
- Comprehensive statistics and logging
- Handles contamination better
- More sensitive for low-quality reads

Bowtie2-only advantages:
- Much faster (3-5x speed improvement)
- Lower memory usage
- Good for high-quality data
- Direct control over alignment parameters

BBMap (honorable mention):
- bbsplit.sh can remove multiple host genomes simultaneously
- Very fast and memory efficient
- Good for complex contamination scenarios

EOF 
