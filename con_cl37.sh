#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 input_file output_file"
    exit 1
fi

input_file=$1
output_file=$2

# Process file with awk to create consensus sequence
awk -F: '
BEGIN {
    max_length = 0;
}
{
    seq = $1;
    count = $2 + 0;  # Convert to number to handle potential leading/trailing spaces
    len = length(seq);
    
    if (len > max_length) {
        max_length = len;
    }
    
    for (i = 1; i <= len; i++) {
        nucleotide = substr(seq, i, 1);
        counts[i, nucleotide] += count;
        # Track if there are any non-standard nucleotides at this position
        if (nucleotide != "A" && nucleotide != "C" && nucleotide != "G" && nucleotide != "T" && nucleotide != "N") {
            has_non_std[i] = 1;
        }
    }
}
END {
    consensus = "";
    for (i = 1; i <= max_length; i++) {
        max_count = 0;
        max_nuc = "N";  # Default to "N" if no nucleotides are found
        
        # Check standard nucleotides first (most common in biological sequences)
        if (counts[i, "A"] > max_count) {
            max_count = counts[i, "A"];
            max_nuc = "A";
        }
        if (counts[i, "C"] > max_count) {
            max_count = counts[i, "C"];
            max_nuc = "C";
        }
        if (counts[i, "G"] > max_count) {
            max_count = counts[i, "G"];
            max_nuc = "G";
        }
        if (counts[i, "T"] > max_count) {
            max_count = counts[i, "T"];
            max_nuc = "T";
        }
        if (counts[i, "N"] > max_count) {
            max_count = counts[i, "N"];
            max_nuc = "N";
        }
        
        # Check non-standard nucleotides only if they exist at this position
        if (has_non_std[i]) {
            for (key in counts) {
                split(key, parts, SUBSEP);
                pos = parts[1];
                nuc = parts[2];
                
                if (pos == i && nuc != "A" && nuc != "C" && nuc != "G" && nuc != "T" && nuc != "N" && counts[pos, nuc] > max_count) {
                    max_count = counts[pos, nuc];
                    max_nuc = nuc;
                }
            }
        }
        
        consensus = consensus max_nuc;
    }
    
    print consensus;
}' "$input_file" > "$output_file"

echo "Consensus sequence written to $output_file"
