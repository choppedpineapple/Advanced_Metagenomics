#!/bin/bash

# =============================================================================
# Antibody IgBLAST Analysis Pipeline
# Process merged FASTQ files for sheep and alpaca antibody samples
# =============================================================================

set -euo pipefail  # Exit on error, undefined variables, pipe failures

# =============================================================================
# CONFIGURATION
# =============================================================================

# Default parameters
THREADS=${THREADS:-8}
SPECIES=""
INPUT_DIR=""
OUTPUT_DIR=""
REFERENCE_DIR=""
MIN_LENGTH=${MIN_LENGTH:-200}
MAX_LENGTH=${MAX_LENGTH:-600}
QUALITY_THRESHOLD=${QUALITY_THRESHOLD:-20}
TRIM_QUALITY=${TRIM_QUALITY:-20}

# Tool paths (modify if tools are not in PATH)
FASTP="fastp"
SEQKIT="seqkit"
IGBLASTN="igblastn"
MAKEBLASTDB="makeblastdb"

# =============================================================================
# FUNCTIONS
# =============================================================================

print_usage() {
    cat << EOF
Usage: $0 -s SPECIES -i INPUT_DIR -o OUTPUT_DIR -r REFERENCE_DIR [OPTIONS]

Required arguments:
    -s SPECIES          Species (sheep or alpaca)
    -i INPUT_DIR        Directory containing merged FASTQ files
    -o OUTPUT_DIR       Output directory for results
    -r REFERENCE_DIR    Directory containing IMGT reference files

Optional arguments:
    -t THREADS          Number of threads (default: 8)
    -l MIN_LENGTH       Minimum sequence length (default: 200)
    -L MAX_LENGTH       Maximum sequence length (default: 600)
    -q QUALITY_THRESHOLD Average quality threshold (default: 20)
    -Q TRIM_QUALITY     Quality trimming threshold (default: 20)
    -h                  Show this help message

Example:
    $0 -s sheep -i ./fastq_files -o ./results -r ./imgt_references -t 16

EOF
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >&2
}

check_dependencies() {
    local missing_tools=()
    
    for tool in $FASTP $SEQKIT $IGBLASTN $MAKEBLASTDB; do
        if ! command -v "$tool" &> /dev/null; then
            missing_tools+=("$tool")
        fi
    done
    
    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_message "ERROR: Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
}

validate_inputs() {
    # Check species
    if [[ "$SPECIES" != "sheep" && "$SPECIES" != "alpaca" ]]; then
        log_message "ERROR: Species must be 'sheep' or 'alpaca'"
        exit 1
    fi
    
    # Check directories
    if [[ ! -d "$INPUT_DIR" ]]; then
        log_message "ERROR: Input directory does not exist: $INPUT_DIR"
        exit 1
    fi
    
    if [[ ! -d "$REFERENCE_DIR" ]]; then
        log_message "ERROR: Reference directory does not exist: $REFERENCE_DIR"
        exit 1
    fi
    
    # Check for FASTQ files
    if ! ls "$INPUT_DIR"/*.fastq* &> /dev/null; then
        log_message "ERROR: No FASTQ files found in $INPUT_DIR"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
}

setup_references() {
    local ref_dir=$1
    local species=$2
    
    log_message "Setting up reference databases for $species"
    
    # Define reference file patterns based on species
    local v_gene_file="${ref_dir}/IGHV_${species}.fasta"
    local d_gene_file="${ref_dir}/IGHD_${species}.fasta"
    local j_gene_file="${ref_dir}/IGHJ_${species}.fasta"
    
    # Check if reference files exist
    for ref_file in "$v_gene_file" "$d_gene_file" "$j_gene_file"; do
        if [[ ! -f "$ref_file" ]]; then
            log_message "WARNING: Reference file not found: $ref_file"
            log_message "Please ensure IMGT reference files are downloaded and named correctly"
        fi
    done
    
    # Create BLAST databases if they don't exist
    for gene_type in V D J; do
        local gene_file="${ref_dir}/IGH${gene_type}_${species}.fasta"
        local db_file="${ref_dir}/IGH${gene_type}_${species}"
        
        if [[ -f "$gene_file" && ! -f "${db_file}.nhr" ]]; then
            log_message "Creating BLAST database for IGH${gene_type}"
            $MAKEBLASTDB -parse_seqids -dbtype nucl -in "$gene_file" -out "$db_file"
        fi
    done
}

process_fastq_file() {
    local input_file=$1
    local output_dir=$2
    local base_name=$(basename "$input_file" | sed 's/\.(fastq\|fq)\(\.gz\)\?$//')
    
    log_message "Processing $base_name"
    
    # Step 1: Quality control and trimming with fastp
    local qc_output="${output_dir}/${base_name}_qc.fastq"
    local fastp_json="${output_dir}/${base_name}_fastp.json"
    local fastp_html="${output_dir}/${base_name}_fastp.html"
    
    $FASTP \
        --in1 "$input_file" \
        --out1 "$qc_output" \
        --json "$fastp_json" \
        --html "$fastp_html" \
        --qualified_quality_phred $QUALITY_THRESHOLD \
        --cut_front \
        --cut_tail \
        --cut_window_size 4 \
        --cut_mean_quality $TRIM_QUALITY \
        --length_required $MIN_LENGTH \
        --length_limit $MAX_LENGTH \
        --low_complexity_filter \
        --complexity_threshold 30 \
        --thread $THREADS \
        --verbose
    
    # Step 2: Additional filtering with seqkit
    local filtered_output="${output_dir}/${base_name}_filtered.fastq"
    
    $SEQKIT seq \
        --min-len $MIN_LENGTH \
        --max-len $MAX_LENGTH \
        --threads $THREADS \
        "$qc_output" > "$filtered_output"
    
    # Step 3: Convert to FASTA for IgBLAST
    local fasta_output="${output_dir}/${base_name}_filtered.fasta"
    
    $SEQKIT fq2fa \
        --threads $THREADS \
        "$filtered_output" > "$fasta_output"
    
    # Step 4: Run IgBLAST
    local igblast_output="${output_dir}/${base_name}_igblast.tsv"
    
    run_igblast "$fasta_output" "$igblast_output" "$SPECIES" "$REFERENCE_DIR"
    
    # Clean up intermediate files (optional)
    rm -f "$qc_output" "$filtered_output"
    
    log_message "Completed processing $base_name"
}

run_igblast() {
    local input_fasta=$1
    local output_file=$2
    local species=$3
    local ref_dir=$4
    
    log_message "Running IgBLAST analysis"
    
    # Define reference database paths
    local v_db="${ref_dir}/IGHV_${species}"
    local d_db="${ref_dir}/IGHD_${species}"
    local j_db="${ref_dir}/IGHJ_${species}"
    
    # Run IgBLAST with comprehensive output format
    $IGBLASTN \
        -query "$input_fasta" \
        -germline_db_V "$v_db" \
        -germline_db_D "$d_db" \
        -germline_db_J "$j_db" \
        -organism "other" \
        -domain_system imgt \
        -ig_seqtype Ig \
        -auxiliary_data optional_file/human_gl.aux \
        -show_translation \
        -outfmt "7 std qseq sseq btop" \
        -num_threads $THREADS \
        -out "$output_file"
    
    # Convert to proper TSV format if needed
    if [[ -s "$output_file" ]]; then
        # Create a proper TSV header and format the output
        local formatted_output="${output_file%.tsv}_formatted.tsv"
        
        {
            echo -e "query_id\tsubject_id\tpercent_identity\talignment_length\tmismatches\tgap_opens\tq_start\tq_end\ts_start\ts_end\tevalue\tbit_score\tquery_seq\tsubject_seq\tbtop"
            grep -v "^#" "$output_file" || true
        } > "$formatted_output"
        
        mv "$formatted_output" "$output_file"
    fi
}

generate_summary_report() {
    local output_dir=$1
    local summary_file="${output_dir}/pipeline_summary.txt"
    
    {
        echo "IgBLAST Pipeline Summary"
        echo "========================"
        echo "Date: $(date)"
        echo "Species: $SPECIES"
        echo "Input directory: $INPUT_DIR"
        echo "Output directory: $OUTPUT_DIR"
        echo "Reference directory: $REFERENCE_DIR"
        echo "Threads used: $THREADS"
        echo "Min length: $MIN_LENGTH"
        echo "Max length: $MAX_LENGTH"
        echo "Quality threshold: $QUALITY_THRESHOLD"
        echo ""
        echo "Processed files:"
        for file in "$output_dir"/*_igblast.tsv; do
            if [[ -f "$file" ]]; then
                local base_name=$(basename "$file" "_igblast.tsv")
                local line_count=$(wc -l < "$file")
                echo "  $base_name: $((line_count - 1)) sequences analyzed"
            fi
        done
    } > "$summary_file"
    
    log_message "Summary report generated: $summary_file"
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

main() {
    log_message "Starting IgBLAST analysis pipeline"
    
    # Parse command line arguments
    while getopts "s:i:o:r:t:l:L:q:Q:h" opt; do
        case $opt in
            s) SPECIES="$OPTARG" ;;
            i) INPUT_DIR="$OPTARG" ;;
            o) OUTPUT_DIR="$OPTARG" ;;
            r) REFERENCE_DIR="$OPTARG" ;;
            t) THREADS="$OPTARG" ;;
            l) MIN_LENGTH="$OPTARG" ;;
            L) MAX_LENGTH="$OPTARG" ;;
            q) QUALITY_THRESHOLD="$OPTARG" ;;
            Q) TRIM_QUALITY="$OPTARG" ;;
            h) print_usage; exit 0 ;;
            \?) echo "Invalid option -$OPTARG" >&2; print_usage; exit 1 ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$SPECIES" || -z "$INPUT_DIR" || -z "$OUTPUT_DIR" || -z "$REFERENCE_DIR" ]]; then
        log_message "ERROR: Missing required arguments"
        print_usage
        exit 1
    fi
    
    # Setup and validation
    check_dependencies
    validate_inputs
    setup_references "$REFERENCE_DIR" "$SPECIES"
    
    # Create subdirectories
    mkdir -p "${OUTPUT_DIR}/qc_reports"
    mkdir -p "${OUTPUT_DIR}/processed_sequences"
    mkdir -p "${OUTPUT_DIR}/igblast_results"
    
    # Process each FASTQ file
    local file_count=0
    for fastq_file in "$INPUT_DIR"/*.fastq*; do
        if [[ -f "$fastq_file" ]]; then
            process_fastq_file "$fastq_file" "${OUTPUT_DIR}/processed_sequences"
            ((file_count++))
        fi
    done
    
    if [[ $file_count -eq 0 ]]; then
        log_message "ERROR: No FASTQ files found to process"
        exit 1
    fi
    
    # Move results to organized directories
    mv "${OUTPUT_DIR}/processed_sequences"/*_fastp.* "${OUTPUT_DIR}/qc_reports/" 2>/dev/null || true
    mv "${OUTPUT_DIR}/processed_sequences"/*_igblast.tsv "${OUTPUT_DIR}/igblast_results/" 2>/dev/null || true
    
    # Generate summary report
    generate_summary_report "${OUTPUT_DIR}/igblast_results"
    
    log_message "Pipeline completed successfully!"
    log_message "Results available in: $OUTPUT_DIR"
    log_message "IgBLAST TSV files in: ${OUTPUT_DIR}/igblast_results/"
}

# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
