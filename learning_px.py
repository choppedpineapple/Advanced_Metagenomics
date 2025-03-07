#!/usr/bin/env python3
"""
Bioinformatics Python Fundamentals Script
Covers: File I/O, Data Structures, Functions, Parallel Processing, and more
"""

# --- Module Imports with Selective Import ---
import os
import gzip
import shutil
from Bio import SeqIO  # Biopython for bioinformatics file handling
import numpy as np
import pandas as pd
from multiprocessing import Pool
from pathlib import Path

# --- Set Working Directory ---
current_dir = Path.cwd()  # Get current working directory
os.chdir(current_dir)  # Explicitly set working directory

# --- Basic Functions ---
def dna_complement(sequence):
    """Return DNA complement with input validation"""
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
    return ''.join([complement.get(base, 'N') for base in sequence])

# --- Multi-Parameter Function ---
def sequence_analyzer(seq, min_length=50, check_gc=True):
    """Analyze DNA sequences with multiple parameters"""
    analysis = {
        'length': len(seq),
        'gc_content': (seq.count('G') + seq.count('C'))/len(seq)*100 if check_gc else None
    }
    return analysis if analysis['length'] >= min_length else None

# --- Loop Structures ---
def process_fastq(file_path):
    """Demonstrate different loop types with FASTQ processing"""
    # For loop with enumerate
    records = list(SeqIO.parse(file_path, "fastq"))
    for i, record in enumerate(records[:3]):  # Process first 3 records
        print(f"Record {i+1}: {record.id}")
    
    # While loop for quality control
    low_qual = 0
    idx = 0
    while idx < len(records):
        if np.mean(records[idx].letter_annotations["phred_quality"]) < 20:
            low_qual += 1
        idx += 1
    print(f"Low quality sequences: {low_qual}")
    
    # Nested loop for k-mer counting
    k = 3
    kmer_counts = {}
    for record in records:
        seq = str(record.seq)
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] = kmer_counts.get(kmer, 0) + 1
    return kmer_counts

# --- File Handling ---
def handle_compressed_files(input_path, output_dir):
    """Process both regular and gzipped FASTQ files"""
    # Check file type
    if input_path.endswith('.gz'):
        with gzip.open(input_path, 'rt') as f:  # Read gzipped file
            records = list(SeqIO.parse(f, 'fastq'))
        new_path = Path(output_dir) / (Path(input_path).stem + '_processed.fastq.gz')
        with gzip.open(new_path, 'wt') as f:  # Write to compressed file
            SeqIO.write(records, f, 'fastq')
    else:
        records = list(SeqIO.parse(input_path, 'fastq'))
        new_path = Path(output_dir) / (Path(input_path).stem + '_processed.fastq')
        with open(new_path, 'w') as f:
            SeqIO.write(records, f, 'fastq')
    return new_path

# --- Recursive Function ---
def generate_permutations(bases, length=3):
    """Recursively generate all DNA sequence permutations"""
    if length == 1:
        return list(bases)
    return [base + perm 
            for base in bases 
            for perm in generate_permutations(bases, length-1)]

# --- Array Operations with NumPy ---
def calculate_quality_stats(qual_scores):
    """Demonstrate NumPy array operations"""
    arr = np.array(qual_scores)
    return {
        'mean': np.mean(arr),
        'std': np.std(arr),
        'max': np.max(arr),
        'min': np.min(arr)
    }

# --- DataFrames with Pandas ---
def create_sequence_df(records):
    """Create pandas DataFrame from sequence records"""
    data = [{
        'id': rec.id,
        'length': len(rec),
        'gc_content': (rec.seq.count('G') + rec.seq.count('C'))/len(rec)*100
    } for rec in records]
    return pd.DataFrame(data)

# --- Parallel Processing ---
def parallel_processing(file_list, num_workers=4):
    """Process multiple files in parallel"""
    with Pool(num_workers) as pool:
        results = pool.map(process_fastq, file_list)
    return results

# --- Main Workflow ---
if __name__ == "__main__":
    # Create test directory
    test_dir = current_dir / 'bioinformatics_test'
    test_dir.mkdir(exist_ok=True)
    
    # Example FASTQ file handling
    sample_fastq = test_dir / 'sample.fastq.gz'
    with gzip.open(sample_fastq, 'wt') as f:
        f.write("@TEST\nACGT\n+\n!!!!\n")
    
    # Process files
    processed_file = handle_compressed_files(sample_fastq, test_dir)
    
    # Demonstrate DataFrame usage
    records = list(SeqIO.parse(gzip.open(sample_fastq, 'rt'), 'fastq'))
    df = create_sequence_df(records)
    print("\nSequence DataFrame:")
    print(df)
    
    # Cleanup
    if processed_file.exists():
        processed_file.unlink()
    if sample_fastq.exists():
        sample_fastq.unlink()
    test_dir.rmdir()

    # Show recursive permutations
    print("\nDNA 3-mers:", generate_permutations('ATGC')[:10])

# --- Key Concepts Covered ---
"""
1. File I/O: Handles both regular and gzipped files using Biopython
2. Functions: Includes simple functions and multi-parameter functions
3. Loops: Demonstrates for, while, and nested loops
4. Data Structures: Uses lists, dictionaries, NumPy arrays, and pandas DataFrames
5. Parallel Processing: Implements multiprocessing with Pool
6. Error Handling: Implicit in file operations (Pathlib)
7. Package Management: Selective imports and function-specific imports
8. Directory Management: Creates/removes directories and files
9. Bioinformatics-Specific: 
   - Sequence handling
   - Quality score analysis
   - k-mer counting
   - DNA permutations
   - GC content calculation
"""
