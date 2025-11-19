import argparse
import sys
import os

def get_args():
    parser = argparse.ArgumentParser(
        description="Extract scFv sequences based on a specific linker sequence from a FASTA file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('-i', '--input', required=True, help="Input Contigs FASTA file")
    parser.add_argument('-o', '--output', required=True, help="Output FASTA file")
    parser.add_argument('-l', '--linker', required=True, help="The 41bp Linker sequence (5'->3')")
    parser.add_argument('-m', '--mismatches', type=int, default=2, help="Maximum allowed mismatches (default: 2)")
    return parser.parse_args()

# --- Helper Functions ---

def reverse_complement(seq):
    """Returns the reverse complement of a DNA sequence."""
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 
                  'a': 't', 'c': 'g', 'g': 'c', 't': 'a', 'N': 'N', 'n': 'n'}
    return "".join(complement.get(base, base) for base in reversed(seq))

def hamming_distance(s1, s2):
    """Calculates number of mismatches between two equal length strings."""
    if len(s1) != len(s2):
        return float('inf')
    return sum(ch1 != ch2 for ch1, ch2 in zip(s1, s2))

def find_motifs(seq, motif, max_mismatches):
    """
    Finds all occurrences of a motif (and its RevComp) in a sequence.
    Returns a list of tuples: (start_index, orientation)
    orientation is 1 for Forward, -1 for Reverse Complement.
    """
    seq_len = len(seq)
    motif_len = len(motif)
    matches = []
    
    motif_fwd = motif
    motif_rev = reverse_complement(motif)
    
    # Optimization: Iterate through sequence
    # Note: For extremely large contigs, Regex might be faster, 
    # but this is robust and dependency-free.
    for i in range(seq_len - motif_len + 1):
        subseq = seq[i : i + motif_len]
        
        # Check Forward
        if hamming_distance(subseq, motif_fwd) <= max_mismatches:
            matches.append((i, 1))
            continue # Assume sequences don't overlap with opposite orientation at same spot
            
        # Check Reverse
        if hamming_distance(subseq, motif_rev) <= max_mismatches:
            matches.append((i, -1))
            
    return matches

def read_fasta(file_path):
    """Generator to read FASTA file without external dependencies."""
    name, seq = None, []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            if line.startswith(">"):
                if name: yield (name, "".join(seq))
                name, seq = line[1:], []
            else:
                seq.append(line)
        if name: yield (name, "".join(seq))

# --- Main Logic ---

def main():
    args = get_args()
    
    input_file = args.input
    output_file = args.output
    linker_seq = args.linker.upper()
    max_mm = args.mismatches
    linker_len = len(linker_seq)
    
    print(f"--- Starting scFv Extraction ---")
    print(f"Input: {input_file}")
    print(f"Linker: {linker_seq} ({linker_len} bps)")
    print(f"Max Mismatches: {max_mm}")
    
    count_extracted = 0
    
    with open(output_file, 'w') as f_out:
        for header, sequence in read_fasta(input_file):
            sequence = sequence.upper()
            seq_len = len(sequence)
            
            # 1. Find matches
            matches = find_motifs(sequence, linker_seq, max_mm)
            
            if not matches:
                continue
                
            # 2. Process matches
            # If a contig has multiple linkers, we treat each as a potential scFv
            for idx, (start_pos, orientation) in enumerate(matches):
                
                # Logic Branch A: Sequence is already within the "Perfect" range (700-950)
                # The prompt implies: if in range, write directly (keep edges),
                # UNLESS the orientation is reversed (Bio implication: we want VH-L-VL).
                
                is_perfect_len = 700 <= seq_len <= 950
                
                final_seq = ""
                extraction_type = ""
                
                if is_perfect_len:
                    # Take the whole contig
                    final_seq = sequence
                    extraction_type = "FULL_CONTIG"
                    
                    # If the linker was found in reverse orientation, we should probably 
                    # reverse complement the whole contig so the output is standardized VH-L-VL
                    if orientation == -1:
                        final_seq = reverse_complement(final_seq)
                        
                else:
                    # Logic Branch B: Trimming (Longer contigs or weird assemblies)
                    # Define window: 400 upstream -- LINKER -- 400 downstream
                    
                    cut_upstream = 400
                    cut_downstream = 400
                    
                    linker_end = start_pos + linker_len
                    
                    # Calculate raw slice indices
                    slice_start = start_pos - cut_upstream
                    slice_end = linker_end + cut_downstream
                    
                    # Handle Boundaries (Assembly imperfections)
                    # If slice_start is negative, it becomes 0 (start of contig)
                    actual_start = max(0, slice_start)
                    # If slice_end is beyond length, it becomes length
                    actual_end = min(seq_len, slice_end)
                    
                    extracted_segment = sequence[actual_start:actual_end]
                    
                    # Check if extraction is viable (e.g. if we only got 50bp total, skip)
                    if len(extracted_segment) < 500:
                        continue 

                    final_seq = extracted_segment
                    extraction_type = "TRIMMED"

                    # Orientation Handling for Trimmed sequences
                    if orientation == -1:
                        final_seq = reverse_complement(final_seq)

                # Final Write
                # Create a unique header
                orient_str = "FWD" if orientation == 1 else "REV"
                new_header = f">{header}_scFv_{idx+1}_{extraction_type}_{orient_str}_len{len(final_seq)}"
                
                f_out.write(f"{new_header}\n{final_seq}\n")
                count_extracted += 1

    print(f"--- Finished ---")
    print(f"Total sequences extracted: {count_extracted}")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    main()
