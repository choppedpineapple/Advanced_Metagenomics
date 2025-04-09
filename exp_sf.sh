#!/bin/bash
# 1. Initialize variables
input="sorted_sheep_light.txt"   # Input file name
output="sheep_light_con.txt"      # Output file name
min_count=20                      # Minimum sequences needed to continue

> "$output"                       # 2. Empty the output file
current="$input"                  # 3. Start processing with original input

# 4. Main processing loop
while [ -s "$current" ]; do       # While current file has data
    # 5. Find highest-count sequence
    ref_line=$(sort -t: -k2nr "$current" | head -n1)
    # Example: From "ATGC:1000" and "GCTA:800", picks "ATGC:1000"
    
    ref_seq=${ref_line%:*}        # 6. Extract sequence (left of colon)
    ref_count=${ref_line#*:}      # Extract count (right of colon)
    
    # 7. Stop if below minimum count
    [ "$ref_count" -lt "$min_count" ] && break
    
    # 8. Set mismatch allowance
    if   [ "$ref_count" -lt 400 ]; then n=3
    elif [ "$ref_count" -lt 1000 ]; then n=4
    else n=5; fi
    
    # 9. Process all sequences
    failed=""                     # Reset failed list
    while IFS= read -r line; do   # Read each line from current file
        seq=${line%:*}            # Extract sequence part
        count=${line#*:}          # Extract count part
        
        # 10. Compare lengths
        if [ ${#seq} -eq ${#ref_seq} ]; then
            # 11. Count mismatches
            mismatch=0
            for i in $(seq 0 $((${#seq}-1))); do
                # Compare each character position
                [ "${seq:$i:1}" != "${ref_seq:$i:1}" ] && ((mismatch++))
            done
            
            # 12. Decision: keep or fail
            if [ $mismatch -le $n ]; then
                echo "${line}_maxkey${ref_count}" >> "$output"
            else
                failed+="$line"$'\n'  # Add to failed list
            fi
        else
            failed+="$line"$'\n'      # Different length = automatic fail
        fi
    done < "$current"
    
    # 13. Create new input for next iteration
    current=$(mktemp)               # Make temporary file
    echo -n "$failed" | sort > "$current"  # Sorted failed sequences
done
