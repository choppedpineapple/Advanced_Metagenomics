#!/bin/bash

# Host read removal using minimap2 (optimized for long reads and fast alignment)
# Usage: ./minimap2_dehost.sh <R1.fastq> <R2.fastq> <output_prefix> <human_ref.fasta>

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

# Set number of threads
THREADS=8

echo "Starting minimap2 host read removal pipeline..."
echo "R1 input: $R1_INPUT"
echo "R2 input: $R2_INPUT"
echo "Output prefix: $OUTPUT_PREFIX"
echo "Human reference: $HUMAN_REF"
echo "Threads: $THREADS"

# Function for paired-end short reads (Illumina)
minimap2_paired_short() {
    echo "=== Minimap2 for paired-end short reads (sr preset) ==="
    
    # Align with sr preset for short reads
    SAM_FILE="${OUTPUT_PREFIX}_aligned.sam"
    minimap2 -ax sr -t "$THREADS" "$HUMAN_REF" "$R1_INPUT" "$R2_INPUT" > "$SAM_FILE"
    
    # Convert to BAM and sort
    BAM_FILE="${OUTPUT_PREFIX}_aligned_sorted.bam"
    samtools view -bS "$SAM_FILE" | samtools sort -@ "$THREADS" -o "$BAM_FILE"
    
    # Extract unmapped reads (both reads in pair unmapped)
    UNMAPPED_BAM="${OUTPUT_PREFIX}_unmapped.bam"
    samtools view -b -f 12 -F 256 "$BAM_FILE" > "$UNMAPPED_BAM"
    
    # Convert back to FASTQ
    OUTPUT_R1="${OUTPUT_PREFIX}_clean_R1.fastq"
    OUTPUT_R2="${OUTPUT_PREFIX}_clean_R2.fastq"
    
    samtools sort -n -@ "$THREADS" "$UNMAPPED_BAM" | \
    samtools fastq -1 "$OUTPUT_R1" -2 "$OUTPUT_R2" -0 /dev/null -s /dev/null -n
    
    # Compress output
    gzip "$OUTPUT_R1" "$OUTPUT_R2"
    
    # Clean up
    rm -f "$SAM_FILE" "$BAM_FILE" "$UNMAPPED_BAM"
    
    echo "Clean reads saved as: ${OUTPUT_R1}.gz ${OUTPUT_R2}.gz"
}

# Function for long reads or noisy short reads (map-ont preset)
minimap2_sensitive() {
    echo "=== Minimap2 sensitive mode for noisy/long reads (map-ont preset) ==="
    
    # Concatenate paired reads for long-read-like processing
    COMBINED_INPUT="${OUTPUT_PREFIX}_combined.fastq"
    cat "$R1_INPUT" "$R2_INPUT" > "$COMBINED_INPUT"
    
    # Align with map-ont preset (more sensitive)
    SAM_FILE="${OUTPUT_PREFIX}_sensitive_aligned.sam"
    minimap2 -ax map-ont -t "$THREADS" "$HUMAN_REF" "$COMBINED_INPUT" > "$SAM_FILE"
    
    # Extract unmapped reads
    UNMAPPED_FASTQ="${OUTPUT_PREFIX}_unmapped_combined.fastq"
    samtools view -f 4 "$SAM_FILE" | awk '{print "@"$1"\n"$10"\n+\n"$11}' > "$UNMAPPED_FASTQ"
    
    # Split back into R1 and R2 based on read names
    # This assumes standard Illumina naming convention (/1, /2 or _1, _2)
    OUTPUT_R1="${OUTPUT_PREFIX}_sensitive_clean_R1.fastq"
    OUTPUT_R2="${OUTPUT_PREFIX}_sensitive_clean_R2.fastq"
    
    # Split reads back to R1/R2
    awk '
    /^@.*[/_]1/ { r1=1; r2=0 }
    /^@.*[/_]2/ { r1=0; r2=1 }
    /^@/ && !/[/_][12]/ { 
        if (NR % 8 <= 4) { r1=1; r2=0 } 
        else { r1=0; r2=1 } 
    }
    r1 { print > "'$OUTPUT_R1'" }
    r2 { print > "'$OUTPUT_R2'" }
    ' "$UNMAPPED_FASTQ"
    
    # Compress output
    gzip "$OUTPUT_R1" "$OUTPUT_R2"
    
    # Clean up
    rm -f "$SAM_FILE" "$COMBINED_INPUT" "$UNMAPPED_FASTQ"
    
    echo "Sensitive mode clean reads saved as: ${OUTPUT_R1}.gz ${OUTPUT_R2}.gz"
}

# Function for ultra-fast screening (asm5 preset)
minimap2_ultrafast() {
    echo "=== Minimap2 ultra-fast screening mode (asm5 preset) ==="
    
    # Use asm5 preset for very fast, less sensitive alignment
    PAF_FILE="${OUTPUT_PREFIX}_aligned.paf"
    
    # Create combined input for single-end-like processing
    COMBINED_INPUT="${OUTPUT_PREFIX}_combined.fastq"
    cat "$R1_INPUT" "$R2_INPUT" > "$COMBINED_INPUT"
    
    # Align and output PAF format (faster than SAM)
    minimap2 -x asm5 -t "$THREADS" "$HUMAN_REF" "$COMBINED_INPUT" > "$PAF_FILE"
    
    # Extract read names that mapped to human
    MAPPED_READS="${OUTPUT_PREFIX}_mapped_reads.txt"
    awk '{print $1}' "$PAF_FILE" | sort -u > "$MAPPED_READS"
    
    # Filter original FASTQ files to remove mapped reads
    OUTPUT_R1="${OUTPUT_PREFIX}_ultrafast_clean_R1.fastq"
    OUTPUT_R2="${OUTPUT_PREFIX}_ultrafast_clean_R2.fastq"
    
    # Use seqtk to exclude mapped reads
    if command -v seqtk &> /dev/null; then
        seqtk subseq "$R1_INPUT" <(comm -23 <(grep '^@' "$R1_INPUT" | sed 's/^@//' | sed 's/[[:space:]].*//' | sort) <(sort "$MAPPED_READS")) > "$OUTPUT_R1"
        seqtk subseq "$R2_INPUT" <(comm -23 <(grep '^@' "$R2_INPUT" | sed 's/^@//' | sed 's/[[:space:]].*//' | sort) <(sort "$MAPPED_READS")) > "$OUTPUT_R2"
    else
        echo "Warning: seqtk not found. Using awk method (slower)"
        # Alternative awk method
        awk 'NR==FNR{mapped[$1]=1; next} /^@/{read=substr($1,2); gsub(/[[:space:]].*/, "", read)} mapped[read]{skip=4} skip>0{skip--; next} 1' "$MAPPED_READS" "$R1_INPUT" > "$OUTPUT_R1"
        awk 'NR==FNR{mapped[$1]=1; next} /^@/{read=substr($1,2); gsub(/[[:space:]].*/, "", read)} mapped[read]{skip=4} skip>0{skip--; next} 1' "$MAPPED_READS" "$R2_INPUT" > "$OUTPUT_R2"
    fi
    
    # Compress output
    gzip "$OUTPUT_R1" "$OUTPUT_R2"
    
    # Clean up
    rm -f "$PAF_FILE" "$COMBINED_INPUT" "$MAPPED_READS"
    
    echo "Ultra-fast clean reads saved as: ${OUTPUT_R1}.gz ${OUTPUT_R2}.gz"
}

# Generate statistics function
generate_stats() {
    local original_r1=$1
    local clean_r1=$2
    local method=$3
    
    STATS_FILE="${OUTPUT_PREFIX}_${method}_removal_stats.txt"
    
    # Count reads (handle both compressed and uncompressed)
    if [[ "$clean_r1" == *.gz ]]; then
        TOTAL_READS=$(zcat "$original_r1" 2>/dev/null | wc -l | awk '{print $1/4}' || wc -l < "$original_r1" | awk '{print $1/4}')
        CLEAN_READS=$(zcat "$clean_r1" | wc -l | awk '{print $1/4}')
    else
        TOTAL_READS=$(wc -l < "$original_r1" | awk '{print $1/4}')
        CLEAN_READS=$(wc -l < "$clean_r1" | awk '{print $1/4}')
    fi
    
    REMOVED_READS=$((TOTAL_READS - CLEAN_READS))
    REMOVAL_PERCENTAGE=$(awk "BEGIN {printf \"%.2f\", ($REMOVED_READS/$TOTAL_READS)*100}")
    
    cat > "$STATS_FILE" << EOF
Minimap2 Host Read Removal Statistics ($method mode)
===========================================
Original read pairs: $TOTAL_READS
Clean read pairs: $CLEAN_READS
Removed read pairs: $REMOVED_READS
Removal percentage: $REMOVAL_PERCENTAGE%

Method: minimap2 $method
Reference genome: $HUMAN_REF
EOF
    
    echo "Statistics written to: $STATS_FILE"
}

# Main execution - choose method based on read characteristics
echo "Choose minimap2 method:"
echo "1) Standard paired-end short reads (sr preset) - recommended for Illumina"
echo "2) Sensitive mode (map-ont preset) - better for noisy or degraded samples"
echo "3) Ultra-fast screening (asm5 preset) - fastest but less sensitive"
echo "4) Run all methods for comparison"

read -p "Enter choice (1-4) [default: 1]: " CHOICE
CHOICE=${CHOICE:-1}

case $CHOICE in
    1)
        minimap2_paired_short
        generate_stats "$R1_INPUT" "${OUTPUT_PREFIX}_clean_R1.fastq.gz" "standard"
        ;;
    2)
        minimap2_sensitive
        generate_stats "$R1_INPUT" "${OUTPUT_PREFIX}_sensitive_clean_R1.fastq.gz" "sensitive"
        ;;
    3)
        minimap2_ultrafast
        generate_stats "$R1_INPUT" "${OUTPUT_PREFIX}_ultrafast_clean_R1.fastq.gz" "ultrafast"
        ;;
    4)
        echo "Running all methods for comparison..."
        minimap2_paired_short
        minimap2_sensitive
        minimap2_ultrafast
        generate_stats "$R1_INPUT" "${OUTPUT_PREFIX}_clean_R1.fastq.gz" "standard"
        generate_stats "$R1_INPUT" "${OUTPUT_PREFIX}_sensitive_clean_R1.fastq.gz" "sensitive"
        generate_stats "$R1_INPUT" "${OUTPUT_PREFIX}_ultrafast_clean_R1.fastq.gz" "ultrafast"
        echo "Comparison completed. Check *_removal_stats.txt files for results."
        ;;
    *)
        echo "Invalid choice. Running standard method."
        minimap2_paired_short
        generate_stats "$R1_INPUT" "${OUTPUT_PREFIX}_clean_R1.fastq.gz" "standard"
        ;;
esac

echo ""
echo "=== Minimap2 Advantages for Metagenomics ==="
cat << 'EOF'

Minimap2 Benefits:
- Extremely fast alignment (faster than BWA MEM and Bowtie2)
- Memory efficient
- Good for both short and long reads
- Multiple presets optimized for different scenarios
- PAF format output option (faster than SAM)

Presets Explained:
- sr: Short single-end reads (Illumina, 454, Sanger)
- map-ont: Oxford Nanopore genomic reads
- asm5: Assembly to reference mapping (very fast, less sensitive)

Speed Comparison (approximate):
1. Minimap2 asm5 preset: Fastest
2. Minimap2 sr preset: Very fast
3. Bowtie2: Fast
4. BWA MEM: Slower
5. Minimap2 map-ont: Most sensitive but slower

Recommendations:
- Use 'sr' preset for standard Illumina metagenomics
- Use 'map-ont' for degraded or contaminated samples
- Use 'asm5' for quick screening of large datasets
- Consider Bowtie2 or KneadData for maximum sensitivity

Installation:
conda install -c bioconda minimap2 samtools seqtk

EOF
