#!/usr/bin/env python3
"""
High-performance FASTQ quality trimmer using dnaio
Trims reads when average quality drops below threshold (default: 20)
"""

import sys
import argparse
from pathlib import Path
import dnaio
import numpy as np
from typing import Optional, Tuple


def phred_to_quality(phred_scores: str) -> np.ndarray:
    """Convert Phred+33 quality scores to numeric values."""
    return np.frombuffer(phred_scores.encode('ascii'), dtype=np.uint8) - 33


def find_trim_position(qualities: np.ndarray, min_avg_quality: int = 20, 
                      window_size: int = 10) -> int:
    """
    Find optimal trim position using sliding window approach.
    Returns the position where to trim (exclusive end).
    """
    if len(qualities) < window_size:
        return len(qualities) if np.mean(qualities) >= min_avg_quality else 0
    
    # Use sliding window to find where quality drops
    for i in range(len(qualities) - window_size + 1):
        window_avg = np.mean(qualities[i:i + window_size])
        if window_avg < min_avg_quality:
            return i
    
    return len(qualities)


def trim_read(sequence: str, quality: str, min_avg_quality: int = 20) -> Tuple[str, str]:
    """
    Trim a single read based on quality scores.
    Returns trimmed sequence and quality strings.
    """
    if not quality:
        return sequence, quality
    
    # Convert quality scores to numeric array
    qual_array = phred_to_quality(quality)
    
    # Find trim position
    trim_pos = find_trim_position(qual_array, min_avg_quality)
    
    # Return trimmed sequences
    return sequence[:trim_pos], quality[:trim_pos]


def process_fastq(input_path: str, output_path: str, min_avg_quality: int = 20,
                 min_length: int = 30, compression_level: int = 6) -> dict:
    """
    Process FASTQ file with quality trimming.
    
    Args:
        input_path: Input FASTQ file path
        output_path: Output FASTQ file path  
        min_avg_quality: Minimum average quality threshold
        min_length: Minimum read length after trimming
        compression_level: Compression level for output (1-9)
    
    Returns:
        Dictionary with processing statistics
    """
    
    stats = {
        'total_reads': 0,
        'trimmed_reads': 0,
        'discarded_reads': 0,
        'kept_reads': 0,
        'total_bases_input': 0,
        'total_bases_output': 0
    }
    
    # Open input and output files with dnaio
    # dnaio automatically handles compression based on file extension
    with dnaio.open(input_path, mode='r') as reader, \
         dnaio.open(output_path, mode='w', compression_level=compression_level) as writer:
        
        for record in reader:
            stats['total_reads'] += 1
            stats['total_bases_input'] += len(record.sequence)
            
            # Trim the read
            trimmed_seq, trimmed_qual = trim_read(
                record.sequence, 
                record.qualities, 
                min_avg_quality
            )
            
            # Check if read was trimmed
            if len(trimmed_seq) < len(record.sequence):
                stats['trimmed_reads'] += 1
            
            # Keep read if it meets minimum length requirement
            if len(trimmed_seq) >= min_length:
                # Create new record with trimmed data
                trimmed_record = dnaio.SequenceRecord(
                    name=record.name,
                    sequence=trimmed_seq,
                    qualities=trimmed_qual
                )
                writer.write(trimmed_record)
                stats['kept_reads'] += 1
                stats['total_bases_output'] += len(trimmed_seq)
            else:
                stats['discarded_reads'] += 1
    
    return stats


def process_paired_fastq(input1_path: str, input2_path: str, 
                        output1_path: str, output2_path: str,
                        min_avg_quality: int = 20, min_length: int = 30,
                        compression_level: int = 6) -> dict:
    """
    Process paired-end FASTQ files with quality trimming.
    Keeps pairs synchronized - if one read is discarded, both are discarded.
    """
    
    stats = {
        'total_pairs': 0,
        'trimmed_pairs': 0,
        'discarded_pairs': 0,
        'kept_pairs': 0,
        'total_bases_input': 0,
        'total_bases_output': 0
    }
    
    with dnaio.open(input1_path, mode='r') as reader1, \
         dnaio.open(input2_path, mode='r') as reader2, \
         dnaio.open(output1_path, mode='w', compression_level=compression_level) as writer1, \
         dnaio.open(output2_path, mode='w', compression_level=compression_level) as writer2:
        
        for record1, record2 in zip(reader1, reader2):
            stats['total_pairs'] += 1
            stats['total_bases_input'] += len(record1.sequence) + len(record2.sequence)
            
            # Trim both reads
            trimmed_seq1, trimmed_qual1 = trim_read(
                record1.sequence, record1.qualities, min_avg_quality
            )
            trimmed_seq2, trimmed_qual2 = trim_read(
                record2.sequence, record2.qualities, min_avg_quality
            )
            
            # Check if reads were trimmed
            if (len(trimmed_seq1) < len(record1.sequence) or 
                len(trimmed_seq2) < len(record2.sequence)):
                stats['trimmed_pairs'] += 1
            
            # Keep pair if both reads meet minimum length requirement
            if len(trimmed_seq1) >= min_length and len(trimmed_seq2) >= min_length:
                # Create new records with trimmed data
                trimmed_record1 = dnaio.SequenceRecord(
                    name=record1.name,
                    sequence=trimmed_seq1,
                    qualities=trimmed_qual1
                )
                trimmed_record2 = dnaio.SequenceRecord(
                    name=record2.name,
                    sequence=trimmed_seq2,
                    qualities=trimmed_qual2
                )
                
                writer1.write(trimmed_record1)
                writer2.write(trimmed_record2)
                stats['kept_pairs'] += 1
                stats['total_bases_output'] += len(trimmed_seq1) + len(trimmed_seq2)
            else:
                stats['discarded_pairs'] += 1
    
    return stats


def print_stats(stats: dict, paired: bool = False) -> None:
    """Print processing statistics."""
    unit = "pairs" if paired else "reads"
    
    print(f"\n=== Processing Statistics ===")
    print(f"Total {unit}: {stats[f'total_{unit[:-1] if unit.endswith("s") else unit}']:,}")
    print(f"Trimmed {unit}: {stats[f'trimmed_{unit[:-1] if unit.endswith("s") else unit}']:,}")
    print(f"Kept {unit}: {stats[f'kept_{unit[:-1] if unit.endswith("s") else unit}']:,}")
    print(f"Discarded {unit}: {stats[f'discarded_{unit[:-1] if unit.endswith("s") else unit}']:,}")
    print(f"Input bases: {stats['total_bases_input']:,}")
    print(f"Output bases: {stats['total_bases_output']:,}")
    
    if stats[f'total_{unit[:-1] if unit.endswith("s") else unit}'] > 0:
        kept_pct = (stats[f'kept_{unit[:-1] if unit.endswith("s") else unit}'] / 
                   stats[f'total_{unit[:-1] if unit.endswith("s") else unit}']) * 100
        print(f"Retention rate: {kept_pct:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Fast FASTQ quality trimmer using dnaio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument("-i", "--input", required=True, 
                       help="Input FASTQ file (or R1 for paired-end)")
    parser.add_argument("-o", "--output", required=True,
                       help="Output FASTQ file (or R1 for paired-end)")
    parser.add_argument("-i2", "--input2", 
                       help="Input R2 FASTQ file for paired-end")
    parser.add_argument("-o2", "--output2",
                       help="Output R2 FASTQ file for paired-end")
    
    # Quality parameters
    parser.add_argument("-q", "--min-quality", type=int, default=20,
                       help="Minimum average quality threshold")
    parser.add_argument("-l", "--min-length", type=int, default=30,
                       help="Minimum read length after trimming")
    
    # Performance parameters
    parser.add_argument("-c", "--compression-level", type=int, default=6,
                       choices=range(1, 10),
                       help="Output compression level (1-9)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.input2 and not args.output2:
        parser.error("--output2 required when --input2 is specified")
    if args.output2 and not args.input2:
        parser.error("--input2 required when --output2 is specified")
    
    # Check input files exist
    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")
    if args.input2 and not Path(args.input2).exists():
        parser.error(f"Input file not found: {args.input2}")
    
    print(f"Processing FASTQ files...")
    print(f"Quality threshold: {args.min_quality}")
    print(f"Minimum length: {args.min_length}")
    
    try:
        if args.input2:  # Paired-end mode
            print(f"Mode: Paired-end")
            print(f"Input: {args.input}, {args.input2}")
            print(f"Output: {args.output}, {args.output2}")
            
            stats = process_paired_fastq(
                args.input, args.input2,
                args.output, args.output2,
                args.min_quality, args.min_length,
                args.compression_level
            )
            print_stats(stats, paired=True)
            
        else:  # Single-end mode
            print(f"Mode: Single-end")
            print(f"Input: {args.input}")
            print(f"Output: {args.output}")
            
            stats = process_fastq(
                args.input, args.output,
                args.min_quality, args.min_length,
                args.compression_level
            )
            print_stats(stats, paired=False)
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    print("Processing completed successfully!")


if __name__ == "__main__":
    main()
