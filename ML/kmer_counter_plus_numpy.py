#!/usr/bin/env python3

import sys
import gzip
import numpy as np

def kmer_to_int(kmer):
    """Convert a kmer string to integer using base-4 encoding (A=0, T=1, G=2, C=3)"""
    base_map = {'A': 0, 'T': 1, 'G': 2, 'C': 3}
    result = 0
    for nucleotide in kmer:
        if nucleotide in base_map:
            result = result * 4 + base_map[nucleotide]
        else:
            # Invalid nucleotide, return -1
            return -1
    return result

def read_fastq(filename):
    """Generator to read FASTQ file (handles both regular and gzipped files)"""
    if filename.endswith('.gz'):
        file_handle = gzip.open(filename, 'rt')
    else:
        file_handle = open(filename, 'r')
    
    try:
        while True:
            # Read 4 lines at a time (FASTQ format)
            header = file_handle.readline().strip()
            if not header:
                break
            sequence = file_handle.readline().strip()
            plus = file_handle.readline().strip()
            quality = file_handle.readline().strip()
            
            yield sequence
    finally:
        file_handle.close()

def extract_kmers(sequence, k):
    """Extract all kmers of size k from a sequence"""
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k].upper()
        # Only include kmers with valid nucleotides
        if all(nucleotide in 'ATGC' for nucleotide in kmer):
            kmers.append(kmer)
    return kmers

def count_kmers(fastq_file, k):
    """Count all kmers of size k in a FASTQ file"""
    kmer_dict = {}
    
    print(f"Processing {fastq_file} with k={k}")
    
    for sequence in read_fastq(fastq_file):
        kmers = extract_kmers(sequence, k)
        for kmer in kmers:
            if kmer in kmer_dict:
                kmer_dict[kmer] += 1
            else:
                kmer_dict[kmer] = 1
    
    return kmer_dict

def main():
    if len(sys.argv) != 3:
        print("Usage: python kmer_counter.py <fastq_file> <kmer_size>")
        print("Example: python kmer_counter.py reads.fastq 5")
        print("Example: python kmer_counter.py reads.fastq.gz 5")
        sys.exit(1)
    
    fastq_file = sys.argv[1]
    try:
        k = int(sys.argv[2])
    except ValueError:
        print("Error: kmer_size must be an integer")
        sys.exit(1)
    
    if k <= 0:
        print("Error: kmer_size must be positive")
        sys.exit(1)
    
    # Count kmers
    kmer_counts = count_kmers(fastq_file, k)
    
    if not kmer_counts:
        print("No valid kmers found!")
        return
    
    # Convert kmers to integers and create numpy arrays
    kmer_strings = list(kmer_counts.keys())
    kmer_integers = []
    counts = []
    
    for kmer in kmer_strings:
        kmer_int = kmer_to_int(kmer)
        if kmer_int != -1:  # Valid kmer
            kmer_integers.append(kmer_int)
            counts.append(kmer_counts[kmer])
    
    # Create numpy arrays
    kmer_array = np.array(kmer_integers, dtype=np.int64)
    count_array = np.array(counts, dtype=np.int32)
    
    # Sort by kmer integer value for consistent output
    sort_indices = np.argsort(kmer_array)
    kmer_array = kmer_array[sort_indices]
    count_array = count_array[sort_indices]
    
    # Display results
    print(f"\nFound {len(kmer_array)} unique kmers of size {k}")
    print(f"Total kmer occurrences: {np.sum(count_array)}")
    
    # Show first 10 kmers as examples
    print("\nFirst 10 kmers (string -> integer -> count):")
    for i in range(min(10, len(kmer_array))):
        # Convert integer back to string for display
        kmer_str = kmer_strings[sort_indices[i]]
        print(f"{kmer_str} -> {kmer_array[i]} -> {count_array[i]}")
    
    # Save results to file
    output_file = f"kmers_k{k}.txt"
    with open(output_file, 'w') as f:
        f.write("kmer_string\tkmer_integer\tcount\n")
        for i in range(len(kmer_array)):
            kmer_str = kmer_strings[sort_indices[i]]
            f.write(f"{kmer_str}\t{kmer_array[i]}\t{count_array[i]}\n")
    
    print(f"\nResults saved to {output_file}")
    print(f"Kmer integers stored in numpy array of shape: {kmer_array.shape}")
    print(f"Count values stored in numpy array of shape: {count_array.shape}")

if __name__ == "__main__":
    main()
