#!/bin/bash

# Help message function
show_help() {
    echo "Usage: $0 [options] <input_file>"
    echo "Generate a consensus sequence from a file containing sequences with counts."
    echo ""
    echo "Options:"
    echo "  -o, --output FILE    Output file (default: consensus.txt)"
    echo "  -t, --threshold NUM  Minimum count threshold (default: 0)"
    echo "  -h, --help           Display this help message and exit"
    echo ""
    echo "Input format expected: SEQUENCE:COUNT (one per line)"
    echo "Example: ATGCCCTTTATATAAANNATTAT:2580"
}

# Default values
OUTPUT_FILE="consensus.txt"
THRESHOLD=0

# Parse command line options
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -t|--threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -*) 
            echo "Unknown option: $1" >&2
            show_help
            exit 1
            ;;
        *)
            INPUT_FILE="$1"
            shift
            ;;
    esac
done

# Check if input file is provided
if [ -z "$INPUT_FILE" ]; then
    echo "Error: Input file is required" >&2
    show_help
    exit 1
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found" >&2
    exit 1
fi

echo "Processing file: $INPUT_FILE"
echo "Output will be written to: $OUTPUT_FILE"
echo "Using count threshold: $THRESHOLD"

# Process the file to create consensus
process_file() {
    local input_file="$1"
    local output_file="$2"
    local threshold="$3"
    
    # Use awk for efficient processing
    awk -F ":" -v threshold="$threshold" '
    BEGIN {
        max_len = 0;
    }
    
    # First pass: find the maximum sequence length and collect all valid sequences
    NR == 1 || NR % 1000000 == 0 { print "Processing line " NR > "/dev/stderr" }
    
    {
        count = $2;
        if (count >= threshold) {
            seq = $1;
            len = length(seq);
            if (len > max_len) max_len = len;
            
            # Store valid sequences and their counts
            seqs[NR] = seq;
            counts[NR] = count;
        }
    }
    
    END {
        print "Maximum sequence length: " max_len > "/dev/stderr";
        print "Building position-specific count matrix..." > "/dev/stderr";
        
        # Initialize count matrices for each position and nucleotide
        for (pos = 1; pos <= max_len; pos++) {
            A[pos] = 0;
            T[pos] = 0;
            G[pos] = 0;
            C[pos] = 0;
            N[pos] = 0;
        }
        
        # Count occurrences of each nucleotide at each position
        for (i in seqs) {
            seq = seqs[i];
            count = counts[i];
            
            for (pos = 1; pos <= length(seq); pos++) {
                nuc = substr(seq, pos, 1);
                if (nuc == "A") A[pos] += count;
                else if (nuc == "T") T[pos] += count;
                else if (nuc == "G") G[pos] += count;
                else if (nuc == "C") C[pos] += count;
                else N[pos] += count;  # Count N or any other characters
            }
        }
        
        # Build consensus sequence
        consensus = "";
        for (pos = 1; pos <= max_len; pos++) {
            max_count = 0;
            max_nuc = "N";
            
            if (A[pos] > max_count) { max_count = A[pos]; max_nuc = "A"; }
            if (T[pos] > max_count) { max_count = T[pos]; max_nuc = "T"; }
            if (G[pos] > max_count) { max_count = G[pos]; max_nuc = "G"; }
            if (C[pos] > max_count) { max_count = C[pos]; max_nuc = "C"; }
            
            # Only add to consensus if we have some data for this position
            if (A[pos] + T[pos] + G[pos] + C[pos] + N[pos] > 0) {
                consensus = consensus max_nuc;
            }
        }
        
        # Output the consensus sequence
        print consensus;
        
        # Also output detailed stats for each position
        print "Position\tA\tT\tG\tC\tN\tConsensus" > "/dev/stderr";
        for (pos = 1; pos <= max_len; pos++) {
            max_count = 0;
            max_nuc = "N";
            
            if (A[pos] > max_count) { max_count = A[pos]; max_nuc = "A"; }
            if (T[pos] > max_count) { max_count = T[pos]; max_nuc = "T"; }
            if (G[pos] > max_count) { max_count = G[pos]; max_nuc = "G"; }
            if (C[pos] > max_count) { max_count = C[pos]; max_nuc = "C"; }
            
            if (A[pos] + T[pos] + G[pos] + C[pos] + N[pos] > 0) {
                print pos "\t" A[pos] "\t" T[pos] "\t" G[pos] "\t" C[pos] "\t" N[pos] "\t" max_nuc > "/dev/stderr";
            }
        }
    }
    ' "$input_file" > "$output_file" 2> "${output_file}.stats"
    
    echo "Consensus sequence written to: $output_file"
    echo "Position statistics written to: ${output_file}.stats"
}

# Run the processing function
process_file "$INPUT_FILE" "$OUTPUT_FILE" "$THRESHOLD"

echo "Done."
