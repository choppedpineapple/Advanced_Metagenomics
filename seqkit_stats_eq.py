#!/usr/bin/env python3
"""
Simple FASTQ statistics script that mimics seqkit stats functionality.
Supports both regular .fastq and compressed .fastq.gz files.
"""

import sys
import gzip
import argparse
from collections import Counter

def open_fastq(filename):
    """Open FASTQ file, handling both regular and gzipped files."""
    if filename.endswith('.gz'):
        return gzip.open(filename, 'rt')
    else:
        return open(filename, 'r')

def parse_fastq(file_handle):
    """
    Parse FASTQ file and yield sequences.
    FASTQ format has 4 lines per record:
    1. Header line starting with @
    2. Sequence line
    3. Plus line starting with +
    4. Quality line
    """
    while True:
        # Read 4 lines at a time
        header = file_handle.readline().strip()
        if not header:  # End of file
            break
        
        sequence = file_handle.readline().strip()
        plus = file_handle.readline().strip()
        quality = file_handle.readline().strip()
        
        # Basic validation
        if not header.startswith('@') or not plus.startswith('+'):
            print(f"Warning: Invalid FASTQ format near header {header}")
            continue
            
        yield sequence

def calculate_stats(sequences):
    """Calculate statistics from a list of sequences."""
    if not sequences:
        return {
            'num_seqs': 0,
            'sum_len': 0,
            'min_len': 0,
            'avg_len': 0,
            'max_len': 0
        }
    
    lengths = [len(seq) for seq in sequences]
    
    return {
        'num_seqs': len(sequences),
        'sum_len': sum(lengths),
        'min_len': min(lengths),
        'avg_len': sum(lengths) / len(lengths),
        'max_len': max(lengths)
    }

def format_number(num):
    """Format large numbers with commas for readability."""
    if isinstance(num, float):
        return f"{num:,.1f}"
    else:
        return f"{num:,}"

def process_fastq_file(filename):
    """Process a single FASTQ file and return statistics."""
    print(f"Processing {filename}...")
    
    sequences = []
    
    try:
        with open_fastq(filename) as f:
            for sequence in parse_fastq(f):
                sequences.append(sequence)
    
    except FileNotFoundError:
        print(f"Error: File {filename} not found")
        return None
    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None
    
    stats = calculate_stats(sequences)
    stats['filename'] = filename
    
    return stats

def print_stats(stats_list):
    """Print statistics in a formatted table."""
    if not stats_list:
        return
    
    # Print header
    print(f"{'file':<20} {'format':<8} {'type':<8} {'num_seqs':<12} {'sum_len':<15} {'min_len':<10} {'avg_len':<10} {'max_len':<10}")
    print("-" * 95)
    
    # Print stats for each file
    for stats in stats_list:
        if stats is None:
            continue
            
        filename = stats['filename']
        file_format = "FASTQ"
        seq_type = "DNA"  # Assuming DNA, could be enhanced to detect
        
        print(f"{filename:<20} {file_format:<8} {seq_type:<8} "
              f"{format_number(stats['num_seqs']):<12} "
              f"{format_number(stats['sum_len']):<15} "
              f"{format_number(stats['min_len']):<10} "
              f"{format_number(stats['avg_len']):<10} "
              f"{format_number(stats['max_len']):<10}")

def main():
    parser = argparse.ArgumentParser(description='Calculate statistics for FASTQ files')
    parser.add_argument('files', nargs='+', help='FASTQ files to process (.fastq or .fastq.gz)')
    
    args = parser.parse_args()
    
    # Process each file
    all_stats = []
    for filename in args.files:
        stats = process_fastq_file(filename)
        all_stats.append(stats)
    
    # Print results
    print_stats(all_stats)

if __name__ == "__main__":
    main()
