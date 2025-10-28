#!/usr/bin/env python3
"""
Multi-threaded FASTQ parser for Python 3.14+ (free-threaded mode)
Handles both regular and gzipped FASTQ files
"""

import sys
import gzip
import threading
from collections import defaultdict
from pathlib import Path


class FASTQRecord:
    """Represents a single FASTQ record"""
    __slots__ = ['header', 'sequence', 'plus', 'quality']
    
    def __init__(self, header, sequence, plus, quality):
        self.header = header
        self.sequence = sequence
        self.plus = plus
        self.quality = quality
    
    def __repr__(self):
        return f"FASTQRecord(id={self.header.split()[0]})"
    
    def to_string(self):
        return f"{self.header}\n{self.sequence}\n{self.plus}\n{self.quality}\n"


def open_fastq(filepath):
    """Open FASTQ file, handling gzip compression automatically"""
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt')
    return open(filepath, 'r')


def parse_fastq_chunk(lines, start_idx, end_idx, results, thread_id):
    """
    Parse a chunk of FASTQ file lines
    
    Args:
        lines: List of all lines from the file
        start_idx: Starting index for this chunk
        end_idx: Ending index for this chunk
        results: Shared dictionary to store results
        thread_id: Thread identifier
    """
    records = []
    
    # Ensure we start at a valid FASTQ record boundary (header line starting with @)
    # Adjust start_idx to the next header if not already at one
    while start_idx < end_idx and not lines[start_idx].startswith('@'):
        start_idx += 1
    
    i = start_idx
    while i < end_idx - 3:  # Need at least 4 lines for a complete record
        # FASTQ format: 4 lines per record
        # Line 1: @ + sequence identifier
        # Line 2: sequence
        # Line 3: + (optionally followed by description)
        # Line 4: quality scores
        
        if not lines[i].startswith('@'):
            i += 1
            continue
            
        header = lines[i].rstrip()
        sequence = lines[i + 1].rstrip()
        plus = lines[i + 2].rstrip()
        quality = lines[i + 3].rstrip()
        
        # Validate the record structure
        if plus.startswith('+') and len(sequence) == len(quality):
            record = FASTQRecord(header, sequence, plus, quality)
            records.append(record)
            i += 4
        else:
            # Skip malformed record
            i += 1
    
    results[thread_id] = records


def parse_fastq_multithreaded(filepath, num_threads=4):
    """
    Parse FASTQ file using multiple threads
    
    Args:
        filepath: Path to FASTQ file (can be gzipped)
        num_threads: Number of threads to use
    
    Returns:
        List of FASTQRecord objects
    """
    print(f"Reading file: {filepath}")
    
    # Read all lines into memory
    with open_fastq(filepath) as f:
        lines = f.readlines()
    
    total_lines = len(lines)
    print(f"Total lines: {total_lines:,}")
    
    # Calculate chunk size for each thread
    chunk_size = total_lines // num_threads
    
    # Shared results dictionary
    results = {}
    threads = []
    
    # Create and start threads
    for i in range(num_threads):
        start_idx = i * chunk_size
        # Last thread takes any remaining lines
        end_idx = total_lines if i == num_threads - 1 else (i + 1) * chunk_size
        
        thread = threading.Thread(
            target=parse_fastq_chunk,
            args=(lines, start_idx, end_idx, results, i)
        )
        threads.append(thread)
        thread.start()
        print(f"Thread {i} started: processing lines {start_idx:,} to {end_idx:,}")
    
    # Wait for all threads to complete
    for i, thread in enumerate(threads):
        thread.join()
        print(f"Thread {i} completed: parsed {len(results[i]):,} records")
    
    # Combine results from all threads
    all_records = []
    for i in range(num_threads):
        all_records.extend(results[i])
    
    return all_records


def calculate_statistics(records):
    """Calculate basic statistics from parsed FASTQ records"""
    if not records:
        return {}
    
    total = len(records)
    seq_lengths = [len(r.sequence) for r in records]
    
    # Calculate GC content
    gc_counts = []
    for record in records:
        seq = record.sequence.upper()
        gc = seq.count('G') + seq.count('C')
        gc_counts.append(gc / len(seq) * 100 if len(seq) > 0 else 0)
    
    stats = {
        'total_records': total,
        'min_length': min(seq_lengths),
        'max_length': max(seq_lengths),
        'avg_length': sum(seq_lengths) / total,
        'avg_gc_content': sum(gc_counts) / total
    }
    
    return stats


def main():
    if len(sys.argv) < 2:
        print("Usage: python fastq_parser.py <fastq_file>")
        print("Example: python fastq_parser.py reads.fastq")
        print("Example: python fastq_parser.py reads.fastq.gz")
        sys.exit(1)
    
    filepath = sys.argv[1]
    
    if not Path(filepath).exists():
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    
    # Parse FASTQ file with 4 threads
    print("\n" + "="*60)
    print("Multi-threaded FASTQ Parser")
    print("="*60 + "\n")
    
    records = parse_fastq_multithreaded(filepath, num_threads=4)
    
    print(f"\n{'='*60}")
    print(f"Parsing complete!")
    print(f"{'='*60}\n")
    
    # Calculate and display statistics
    stats = calculate_statistics(records)
    
    print("Statistics:")
    print(f"  Total records: {stats['total_records']:,}")
    print(f"  Sequence length range: {stats['min_length']} - {stats['max_length']} bp")
    print(f"  Average length: {stats['avg_length']:.1f} bp")
    print(f"  Average GC content: {stats['avg_gc_content']:.2f}%")
    
    # Display first few records as sample
    print(f"\nFirst 3 records:")
    for i, record in enumerate(records[:3], 1):
        print(f"\nRecord {i}:")
        print(f"  Header: {record.header}")
        print(f"  Sequence: {record.sequence[:50]}{'...' if len(record.sequence) > 50 else ''}")
        print(f"  Quality: {record.quality[:50]}{'...' if len(record.quality) > 50 else ''}")


if __name__ == "__main__":
    main()
