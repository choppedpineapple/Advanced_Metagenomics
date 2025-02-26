#!/bin/bash

# ===========================================================================
# dehost.sh - Efficient removal of host (human) reads from paired-end 
# Illumina gut microbiome shotgun sequencing data
# ===========================================================================

set -e
set -o pipefail

# Default values
THREADS=8
MIN_QUALITY=20
MIN_LENGTH=50
ADAPTER_FILE=""
HUMAN_REFERENCE=""
CONTAMINANT_DB=""
OUTPUT_DIR="./dehosted_data"
KEEP_INTERMEDIATE=false
FAST_MODE=false

# Function to display usage
usage() {
    echo "Usage: $0 [options] -1 <forward_reads.fastq.gz> -2 <reverse_reads.fastq.gz>"
    echo
    echo "Options:"
    echo "  -1, --r1 FILE         Forward reads file (required)"
    echo "  -2, --r2 FILE         Reverse reads file (required)"
    echo "  -o, --output DIR      Output directory (default: ./dehosted_data)"
    echo "  -t, --threads INT     Number of threads to use (default: 8)"
    echo "  -q, --quality INT     Minimum quality score for trimming (default: 20)"
    echo "  -l, --length INT      Minimum read length after trimming (default: 50)"
    echo "  -a, --adapter FILE    Adapter sequences file for trimming"
    echo "  -h, --human REF       Human reference genome (default: GRCh38)"
    echo "  -c, --contaminants DB Contaminant database (default: PhiX)"
    echo "  -k, --keep            Keep intermediate files"
    echo "  -f, --fast            Fast mode (less sensitive but faster)" 
    echo "  --help                Display this help message"
    exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -1|--r1)
            R1="$2"
            shift 2
            ;;
        -2|--r2)
            R2="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -t|--threads)
            THREADS="$2"
            shift 2
            ;;
        -q|--quality)
            MIN_QUALITY="$2"
            shift 2
            ;;
        -l|--length)
            MIN_LENGTH="$2"
            shift 2
            ;;
        -a|--adapter)
            ADAPTER_FILE="$2"
            shift 2
            ;;
        -h|--human)
            HUMAN_REFERENCE="$2"
            shift 2
            ;;
        -c|--contaminants)
            CONTAMINANT_DB="$2"
            shift 2
            ;;
        -k|--keep)
            KEEP_INTERMEDIATE=true
            shift
            ;;
        -f|--fast)
            FAST_MODE=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Verify required arguments
if [ -z "$R1" ] || [ -z "$R2" ]; then
    echo "Error: Forward and reverse read files are required."
    usage
fi

# Check if input files exist
if [ ! -f "$R1" ] || [ ! -f "$R2" ]; then
    echo "Error: Input files do not exist."
    exit 1
fi

# Extract sample name from input filename
SAMPLE_NAME=$(basename "$R1" | sed 's/_R1.*//' | sed 's/.R1.*//' | sed 's/_1.*//')
echo "Processing sample: $SAMPLE_NAME"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/tmp"

# Log file
LOG_FILE="$OUTPUT_DIR/logs/${SAMPLE_NAME}_dehost.log"
STATS_FILE="$OUTPUT_DIR/logs/${SAMPLE_NAME}_stats.txt"

# Initialize log file
echo "=== De-hosting pipeline for $SAMPLE_NAME started at $(date) ===" > "$LOG_FILE"
echo "Command: $0 $@" >> "$LOG_FILE"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
for cmd in fastp bowtie2 samtools pigz; do
    if ! command_exists "$cmd"; then
        echo "Error: Required command '$cmd' not found. Please install it before running this script."
        exit 1
    fi
done

# Set up human reference genome if not provided
if [ -z "$HUMAN_REFERENCE" ]; then
    # Check if pre-built Bowtie2 index for GRCh38 exists in common locations
    if [ -f "/usr/local/share/bowtie2-indexes/GRCh38/GRCh38" ]; then
        HUMAN_REFERENCE="/usr/local/share/bowtie2-indexes/GRCh38/GRCh38"
    elif [ -f "$HOME/references/GRCh38/GRCh38" ]; then
        HUMAN_REFERENCE="$HOME/references/GRCh38/GRCh38"
    else
        echo "Warning: Human reference genome not specified and default not found."
        echo "Will continue without human read filtering. Specify with -h option."
        HUMAN_REFERENCE=""
    fi
fi

# Set up contaminant database if not provided
if [ -z "$CONTAMINANT_DB" ]; then
    # Check if PhiX index exists in common locations
    if [ -f "/usr/local/share/bowtie2-indexes/phix/phix" ]; then
        CONTAMINANT_DB="/usr/local/share/bowtie2-indexes/phix/phix"
    elif [ -f "$HOME/references/phix/phix" ]; then
        CONTAMINANT_DB="$HOME/references/phix/phix"
    else
        echo "Warning: Contaminant database not specified and default not found."
        echo "Will continue without contaminant filtering. Specify with -c option."
        CONTAMINANT_DB=""
    fi
fi

# Set adapter file if not provided
if [ -z "$ADAPTER_FILE" ]; then
    # Use fastp's built-in adapter detection
    ADAPTER_PARAM="--detect_adapter_for_pe"
else
    ADAPTER_PARAM="--adapter_fasta $ADAPTER_FILE"
fi

# Temporary files
TRIMMED_R1="$OUTPUT_DIR/tmp/${SAMPLE_NAME}_trimmed_R1.fastq.gz"
TRIMMED_R2="$OUTPUT_DIR/tmp/${SAMPLE_NAME}_trimmed_R2.fastq.gz"
NO_HUMAN_R1="$OUTPUT_DIR/tmp/${SAMPLE_NAME}_no_human_R1.fastq.gz"
NO_HUMAN_R2="$OUTPUT_DIR/tmp/${SAMPLE_NAME}_no_human_R2.fastq.gz"
FINAL_R1="$OUTPUT_DIR/${SAMPLE_NAME}_clean_R1.fastq.gz"
FINAL_R2="$OUTPUT_DIR/${SAMPLE_NAME}_clean_R2.fastq.gz"

# Stats variables
TOTAL_READS=0
READS_AFTER_QC=0
READS_AFTER_HUMAN=0
READS_AFTER_CONTAMINANT=0

echo "Step 1: Quality control and adapter trimming with fastp" | tee -a "$LOG_FILE"
fastp \
    --in1 "$R1" \
    --in2 "$R2" \
    --out1 "$TRIMMED_R1" \
    --out2 "$TRIMMED_R2" \
    --json "$OUTPUT_DIR/logs/${SAMPLE_NAME}_fastp.json" \
    --html "$OUTPUT_DIR/logs/${SAMPLE_NAME}_fastp.html" \
    --thread "$THREADS" \
    --qualified_quality_phred "$MIN_QUALITY" \
    --length_required "$MIN_LENGTH" \
    --correction \
    --overrepresentation_analysis \
    $ADAPTER_PARAM \
    --trim_poly_g \
    --cut_right \
    --cut_window_size 4 \
    --cut_mean_quality 20 \
    --low_complexity_filter \
    2>> "$LOG_FILE"

# Extract stats
TOTAL_READS=$(grep "total_reads" "$OUTPUT_DIR/logs/${SAMPLE_NAME}_fastp.json" | head -1 | awk -F': ' '{print $2}' | tr -d ',')
READS_AFTER_QC=$(grep "after_filtering" -A8 "$OUTPUT_DIR/logs/${SAMPLE_NAME}_fastp.json" | grep "total_reads" | head -1 | awk -F': ' '{print $2}' | tr -d ',')
echo "Initial read count: $TOTAL_READS" > "$STATS_FILE"
echo "Reads after QC: $READS_AFTER_QC" >> "$STATS_FILE"
echo "Reads filtered by QC: $((TOTAL_READS - READS_AFTER_QC)) ($(echo "scale=2; 100 * ($TOTAL_READS - $READS_AFTER_QC) / $TOTAL_READS" | bc)%)" >> "$STATS_FILE"

# Set up mapping arguments based on mode
if [ "$FAST_MODE" = true ]; then
    MAPPING_ARGS="--very-fast --ignore-quals"
else
    MAPPING_ARGS="--sensitive-local"
fi

# Step 2: Remove human reads if reference is available
if [ -n "$HUMAN_REFERENCE" ]; then
    echo "Step 2: Removing human reads with Bowtie2" | tee -a "$LOG_FILE"
    # Map to human genome and keep unmapped reads (non-human)
    bowtie2 \
        -x "$HUMAN_REFERENCE" \
        -1 "$TRIMMED_R1" \
        -2 "$TRIMMED_R2" \
        -p "$THREADS" \
        $MAPPING_ARGS \
        --un-conc-gz "$OUTPUT_DIR/tmp/${SAMPLE_NAME}_no_human_%.fastq.gz" \
        2>> "$LOG_FILE" | \
    samtools view -f 12 -F 256 -b -@ "$THREADS" | \
    samtools sort -n -@ "$THREADS" -o /dev/null
    
    # Rename output files from Bowtie2
    mv "$OUTPUT_DIR/tmp/${SAMPLE_NAME}_no_human_1.fastq.gz" "$NO_HUMAN_R1"
    mv "$OUTPUT_DIR/tmp/${SAMPLE_NAME}_no_human_2.fastq.gz" "$NO_HUMAN_R2"
    
    # Count reads after human filtering
    READS_AFTER_HUMAN=$(zcat "$NO_HUMAN_R1" | echo $((`wc -l`/4)))
    echo "Reads after human filtering: $READS_AFTER_HUMAN" >> "$STATS_FILE"
    echo "Human reads removed: $((READS_AFTER_QC - READS_AFTER_HUMAN)) ($(echo "scale=2; 100 * ($READS_AFTER_QC - $READS_AFTER_HUMAN) / $READS_AFTER_QC" | bc)%)" >> "$STATS_FILE"
else
    echo "Step 2: Skipping human read removal (no reference provided)" | tee -a "$LOG_FILE"
    NO_HUMAN_R1="$TRIMMED_R1"
    NO_HUMAN_R2="$TRIMMED_R2"
    READS_AFTER_HUMAN="$READS_AFTER_QC"
fi

# Step 3: Remove contaminant reads if database is available
if [ -n "$CONTAMINANT_DB" ]; then
    echo "Step 3: Removing contaminant reads with Bowtie2" | tee -a "$LOG_FILE"
    # Map to contaminant database and keep unmapped reads
    bowtie2 \
        -x "$CONTAMINANT_DB" \
        -1 "$NO_HUMAN_R1" \
        -2 "$NO_HUMAN_R2" \
        -p "$THREADS" \
        $MAPPING_ARGS \
        --un-conc-gz "$OUTPUT_DIR/tmp/${SAMPLE_NAME}_clean_%.fastq.gz" \
        2>> "$LOG_FILE" | \
    samtools view -f 12 -F 256 -b -@ "$THREADS" | \
    samtools sort -n -@ "$THREADS" -o /dev/null
    
    # Rename output files from Bowtie2
    mv "$OUTPUT_DIR/tmp/${SAMPLE_NAME}_clean_1.fastq.gz" "$FINAL_R1"
    mv "$OUTPUT_DIR/tmp/${SAMPLE_NAME}_clean_2.fastq.gz" "$FINAL_R2"
    
    # Count reads after contaminant filtering
    READS_AFTER_CONTAMINANT=$(zcat "$FINAL_R1" | echo $((`wc -l`/4)))
    echo "Reads after contaminant filtering: $READS_AFTER_CONTAMINANT" >> "$STATS_FILE"
    echo "Contaminant reads removed: $((READS_AFTER_HUMAN - READS_AFTER_CONTAMINANT)) ($(echo "scale=2; 100 * ($READS_AFTER_HUMAN - $READS_AFTER_CONTAMINANT) / $READS_AFTER_HUMAN" | bc)%)" >> "$STATS_FILE"
else
    echo "Step 3: Skipping contaminant removal (no database provided)" | tee -a "$LOG_FILE"
    cp "$NO_HUMAN_R1" "$FINAL_R1"
    cp "$NO_HUMAN_R2" "$FINAL_R2"
    READS_AFTER_CONTAMINANT="$READS_AFTER_HUMAN"
fi

# Calculate overall statistics
echo "Step 4: Generating final statistics" | tee -a "$LOG_FILE"
TOTAL_REMOVED=$((TOTAL_READS - READS_AFTER_CONTAMINANT))
PERCENT_REMOVED=$(echo "scale=2; 100 * $TOTAL_REMOVED / $TOTAL_READS" | bc)
echo "Total reads removed: $TOTAL_REMOVED ($PERCENT_REMOVED%)" >> "$STATS_FILE"
echo "Final clean reads: $READS_AFTER_CONTAMINANT ($(echo "scale=2; 100 * $READS_AFTER_CONTAMINANT / $TOTAL_READS" | bc)%)" >> "$STATS_FILE"

# Print summary
echo "=== De-hosting Summary for $SAMPLE_NAME ===" | tee -a "$LOG_FILE"
cat "$STATS_FILE" | tee -a "$LOG_FILE"

# Clean up intermediate files if requested
if [ "$KEEP_INTERMEDIATE" = false ]; then
    echo "Cleaning up intermediate files..." | tee -a "$LOG_FILE"
    rm -rf "$OUTPUT_DIR/tmp"
else
    echo "Keeping intermediate files in $OUTPUT_DIR/tmp" | tee -a "$LOG_FILE"
fi

echo "=== De-hosting pipeline completed at $(date) ===" | tee -a "$LOG_FILE"
echo "Clean reads available at:"
echo "$FINAL_R1"
echo "$FINAL_R2"
