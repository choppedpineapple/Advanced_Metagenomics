#!/bin/bash
input="sorted_sheep_light.txt"
output="sheep_light_con.txt"
min_count=20

> "$output"
current="$input"

while [ -s "$current" ]; do
    # Find highest count sequence
    ref_line=$(sort -t: -k2nr "$current" | head -n1)
    ref_seq=${ref_line%:*}
    ref_count=${ref_line#*:}
    
    [ "$ref_count" -lt "$min_count" ] && break
    
    # Set mismatch tolerance
    if   [ "$ref_count" -lt 400 ]; then n=3
    elif [ "$ref_count" -lt 1000 ]; then n=4
    else n=5; fi
    
    # Process all sequences
    failed=""
    while IFS= read -r line; do
        seq=${line%:*}
        if [ ${#seq} -eq ${#ref_seq} ]; then
            mismatch=0
            for i in $(seq 0 $((${#seq}-1))); do
                [ "${seq:$i:1}" != "${ref_seq:$i:1}" ] && ((mismatch++))
            done
            if [ $mismatch -le $n ]; then
                echo "${line}_maxkey${ref_count}" >> "$output"
            else
                failed+="$line"$'\n'
            fi
        else
            failed+="$line"$'\n'
        fi
    done < "$current"
    
    # Sort failed for deterministic next iteration
    current=$(mktemp)
    echo -n "$failed" | sort > "$current"
done
