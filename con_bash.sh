#!/bin/bash

# Default parameters
INPUT_FILE="sorted_sheep_light.txt"
OUTPUT_FILE="sheep_light_con.txt"
THREADS=32
OUTPUT_DIR="/mnt/storage-HDD05a/1.scrach/immuno-mahek/WS1-scrach/crp/result/"
MIN_COUNT=20

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --input FILE         Input file with sequences and counts (default: $INPUT_FILE)"
    echo "  --output FILE        Output file for results (default: $OUTPUT_FILE)"
    echo "  --threads NUM        Number of threads to use (default: $THREADS)"
    echo "  --output-dir DIR     Directory for output files (default: $OUTPUT_DIR)"
    echo "  --min-count NUM      Minimum count to consider a sequence (default: $MIN_COUNT)"
    echo "  --help               Display this help message"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        --threads)
            THREADS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --min-count)
            MIN_COUNT="$2"
            shift 2
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

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Start logging
LOG_FILE="sequence_filtering.log"
echo "$(date) - INFO - Script started" > "$LOG_FILE"

# Function to calculate mismatches between two sequences
calculate_mismatch() {
    local seq1="$1"
    local seq2="$2"
    
    # Check if lengths are equal
    if [ ${#seq1} -ne ${#seq2} ]; then
        echo "9999" # Large number to represent infinity
        return
    fi
    
    # Quick check for identical sequences
    if [ "$seq1" = "$seq2" ]; then
        echo "0"
        return
    fi
    
    # Count mismatches
    local mismatches=0
    local length=${#seq1}
    
    for ((i=0; i<length; i++)); do
        if [ "${seq1:$i:1}" != "${seq2:$i:1}" ]; then
            ((mismatches++))
        fi
    done
    
    echo "$mismatches"
}

# Function to process a chunk of sequences
process_chunk() {
    local chunk_file="$1"
    local ref_sequence="$2"
    local tolerance="$3"
    local max_key="$4"
    local output_dir="$5"
    
    local temp_output="${output_dir}/temp_passed_${max_key}_$$_${RANDOM}.txt"
    local failed_output=$(mktemp)
    
    # Process each line
    while IFS= read -r line; do
        # Extract sequence and count
        seq=$(echo "$line" | cut -d':' -f1)
        
        # Only compare sequences of equal length
        if [ ${#seq} -eq ${#ref_sequence} ]; then
            # Calculate mismatch
            mismatch=$(calculate_mismatch "$seq" "$ref_sequence")
            
            if [ "$mismatch" -le "$tolerance" ]; then
                # Passed - add to output file
                echo "${line}_maxkey${max_key}" >> "$temp_output"
            else
                # Failed - add to return value
                echo "$line" >> "$failed_output"
            fi
        else
            # Length mismatch - failed
            echo "$line" >> "$failed_output"
        fi
    done < "$chunk_file"
    
    # Output failed sequences
    cat "$failed_output"
    rm -f "$failed_output"
}

# Main processing loop
start_time=$(date +%s)
echo "Script started at: $(date)"

# Create a copy of the input file to work with
remain_strings_file=$(mktemp)
cat "$INPUT_FILE" > "$remain_strings_file"

# Create/clear output file
> "$OUTPUT_FILE"

iteration=0
# Continue until no sequences remain
while [ -s "$remain_strings_file" ]; do
    ((iteration++))
    echo "$(date) - INFO - Starting iteration $iteration" >> "$LOG_FILE"
    
    # Count the number of sequences
    seq_count=$(wc -l < "$remain_strings_file")
    echo "$(date) - INFO - Processing $seq_count sequences" >> "$LOG_FILE"
    
    # Find the sequence with the highest count
    highest_line=$(sort -t':' -k2,2nr "$remain_strings_file" | head -n 1)
    max_seq=$(echo "$highest_line" | cut -d':' -f1)
    max_count=$(echo "$highest_line" | cut -d':' -f2)
    
    echo "$(date) - INFO - Reference sequence (count=$max_count): ${max_seq:0:30}..." >> "$LOG_FILE"
    
    # Check if count is below threshold
    if [ "$max_count" -lt "$MIN_COUNT" ]; then
        echo "$(date) - INFO - Max count $max_count is less than minimum threshold $MIN_COUNT. Stopping." >> "$LOG_FILE"
        break
    fi
    
    # Set mismatch tolerance based on count
    if [ "$max_count" -lt 400 ]; then
        tolerance=3
    elif [ "$max_count" -lt 1000 ]; then
        tolerance=4
    else
        tolerance=5
    fi
    
    echo "$(date) - INFO - Using mismatch tolerance: $tolerance" >> "$LOG_FILE"
    
    # Create temporary directory for this iteration
    temp_dir="temp_${max_count}_${seq_count}"
    mkdir -p "$temp_dir"
    
    # Split input data into chunks for parallel processing
    lines_per_file=$(( (seq_count + THREADS - 1) / THREADS ))
    split -l "$lines_per_file" "$remain_strings_file" "${temp_dir}/chunk_"
    
    # Find all chunk files
    chunk_files=("${temp_dir}/chunk_"*)
    echo "$(date) - INFO - Split data into ${#chunk_files[@]} chunks for parallel processing" >> "$LOG_FILE"
    
    # Process chunks in parallel
    failed_file=$(mktemp)
    
    for chunk_file in "${chunk_files[@]}"; do
        process_chunk "$chunk_file" "$max_seq" "$tolerance" "$max_count" "$OUTPUT_DIR" >> "$failed_file" &
        
        # Limit number of parallel processes
        running_jobs=$(jobs -r | wc -l)
        while [ "$running_jobs" -ge "$THREADS" ]; do
            sleep 0.1
            running_jobs=$(jobs -r | wc -l)
        done
    done
    
    # Wait for all background jobs to complete
    wait
    
    # Sort failed sequences for consistent ordering
    sort "$failed_file" > "$remain_strings_file"
    rm -f "$failed_file"
    
    # Clean up temporary directory
    rm -rf "$temp_dir"
    
    echo "$(date) - INFO - Iteration $iteration complete" >> "$LOG_FILE"
    
    # Count remaining sequences
    remain_count=$(wc -l < "$remain_strings_file")
    echo "$(date) - INFO - Remaining sequences: $remain_count" >> "$LOG_FILE"
done

# Combine all temp_passed files into the output file
cat "${OUTPUT_DIR}"/temp_passed_* >> "$OUTPUT_FILE" 2>/dev/null

# Clean up temporary files
rm -f "${OUTPUT_DIR}"/temp_passed_*
rm -f "$remain_strings_file"

# Report execution time
end_time=$(date +%s)
elapsed_time=$((end_time - start_time))
echo "$(date) - INFO - Script completed in $elapsed_time seconds" >> "$LOG_FILE"
echo "Script started at: $(date -d@$start_time)"
echo "Script ended at: $(date -d@$end_time)"
echo "Total execution time: $elapsed_time seconds"
