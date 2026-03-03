#!/usr/bin/env python3

import sys
import regex

def match_with_tolerance(pattern, text, max_mismatches=4):
    fuzzy_pattern = f"({pattern}){{s<={max_mismatches}}}"
    match = regex.search(fuzzy_pattern, text)
    return match is not None, match

def parse_fasta(file_path):
    """A proper FASTA parser so we don't accidentally run regex on our headers."""
    header = None
    seq = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header:
                    yield header, "".join(seq)
                header = line
                seq = []
            else:
                seq.append(line)
        if header:
            yield header, "".join(seq)

def split_scfv(pattern, pattern_rc, input_file, before_file, after_file):
    with open(before_file, 'w') as before_out, open(after_file, 'w') as after_out:
        
        success_count = 0
        fail_count = 0

        for header, text in parse_fasta(input_file):
            is_match, m = match_with_tolerance(pattern, text)
            if is_match:
                start, end = m.span()
                before_seq = text[:start]
                after_seq = text[end:]
                
                # Write EXACT SAME header so Polars can inner join them later
                before_out.write(f"{header}\n{before_seq}\n")
                after_out.write(f"{header}\n{after_seq}\n")
                success_count += 1
                continue

            is_rc_match, n = match_with_tolerance(pattern_rc, text)
            if is_rc_match:
                start, end = n.span()
                before_seq = text[:start]
                after_seq = text[end:]
                
                # If it's reverse complemented, swap the order to maintain biological orientation
                before_out.write(f"{header}\n{after_seq}\n")
                after_out.write(f"{header}\n{before_seq}\n")
                success_count += 1
                continue
            
            # If the regex fails entirely, log it and drop it. 
            fail_count += 1
            print(f"Warning: No linker found for {header}. Dropping it.")

        print(f"\nDone! Successfully split: {success_count} | Failed/Dropped: {fail_count}")

def main():
    if len(sys.argv) != 4:
        print("Usage: python split.py <input.fasta> <before_out.fasta> <after_out.fasta>")
        sys.exit(1)

    pattern = "GAAGGTAAATCTTCTGGCTCTGGCTCTGAGTCTAAAGTGGAT"
    pattern_rc = "ATCCACTTTAGACTCAGAGCCAGAGCCAGAAGATTTACCTTC"
    
    input_file = sys.argv[1]
    before_file = sys.argv[2] 
    after_file = sys.argv[3]  
    
    split_scfv(pattern, pattern_rc, input_file, before_file, after_file)

if __name__ == "__main__":
    main()
          
