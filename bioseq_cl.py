#!/usr/bin/env python3
"""
Comprehensive DNA Sequence Processing Script using Biopython
===========================================================

This script demonstrates various DNA sequence manipulation and quality control
techniques for Illumina short reads (both single-end and paired-end).

Features:
- Quality-based trimming
- Sliding window quality filtering
- Basic sequence statistics
- GC content analysis
- Sequence manipulation functions
- Support for gzipped FASTQ files

Author: Learning Script
Date: 2024
"""

import gzip
import os
import statistics
from typing import List, Tuple, Optional, Dict, Any
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import GC
import argparse


def open_fastq_file(filepath: str):
    """
    Open FASTQ file, handling both gzipped and uncompressed files.
    
    Args:
        filepath (str): Path to the FASTQ file
        
    Returns:
        File handle
    """
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt')
    else:
        return open(filepath, 'r')


def quality_trim_single_read(record: SeqRecord, min_quality: int = 20) -> Optional[SeqRecord]:
    """
    Trim a single read from both ends based on quality scores.
    Removes low-quality bases from 5' and 3' ends.
    
    Args:
        record (SeqRecord): Input sequence record
        min_quality (int): Minimum quality threshold (default: 20)
        
    Returns:
        SeqRecord or None: Trimmed sequence or None if too short
    """
    if not hasattr(record, 'letter_annotations') or 'phred_quality' not in record.letter_annotations:
        print(f"Warning: No quality scores found for {record.id}")
        return record
    
    qualities = record.letter_annotations['phred_quality']
    seq_len = len(record.seq)
    
    # Find 5' trim position
    start_pos = 0
    for i, qual in enumerate(qualities):
        if qual >= min_quality:
            start_pos = i
            break
    else:
        # No bases meet quality threshold
        return None
    
    # Find 3' trim position
    end_pos = seq_len
    for i in range(seq_len - 1, -1, -1):
        if qualities[i] >= min_quality:
            end_pos = i + 1
            break
    
    # Check if remaining sequence is long enough
    if end_pos - start_pos < 30:  # Minimum length threshold
        return None
    
    # Create trimmed record
    trimmed_seq = record.seq[start_pos:end_pos]
    trimmed_qual = qualities[start_pos:end_pos]
    
    trimmed_record = SeqRecord(
        seq=trimmed_seq,
        id=record.id,
        description=f"{record.description} trimmed:{start_pos}-{end_pos}"
    )
    trimmed_record.letter_annotations['phred_quality'] = trimmed_qual
    
    return trimmed_record


def sliding_window_quality_filter(record: SeqRecord, window_size: int = 10, 
                                min_avg_quality: int = 20) -> Optional[SeqRecord]:
    """
    Filter reads using sliding window approach.
    Finds the longest subsequence where average quality in any window >= threshold.
    
    Args:
        record (SeqRecord): Input sequence record
        window_size (int): Size of sliding window (default: 10)
        min_avg_quality (int): Minimum average quality in window (default: 20)
        
    Returns:
        SeqRecord or None: Filtered sequence or None if no good region found
    """
    if not hasattr(record, 'letter_annotations') or 'phred_quality' not in record.letter_annotations:
        return record
    
    qualities = record.letter_annotations['phred_quality']
    seq_len = len(record.seq)
    
    if seq_len < window_size:
        return None
    
    # Find all positions where sliding window meets quality threshold
    good_positions = []
    for i in range(seq_len - window_size + 1):
        window_quals = qualities[i:i + window_size]
        avg_qual = statistics.mean(window_quals)
        if avg_qual >= min_avg_quality:
            good_positions.extend(range(i, i + window_size))
    
    if not good_positions:
        return None
    
    # Find longest continuous stretch of good positions
    good_positions = sorted(set(good_positions))  # Remove duplicates and sort
    
    # Find longest consecutive sequence
    best_start, best_end = 0, 0
    current_start = good_positions[0]
    current_end = good_positions[0]
    
    for i in range(1, len(good_positions)):
        if good_positions[i] == good_positions[i-1] + 1:
            current_end = good_positions[i]
        else:
            if current_end - current_start > best_end - best_start:
                best_start, best_end = current_start, current_end
            current_start = current_end = good_positions[i]
    
    # Check final stretch
    if current_end - current_start > best_end - best_start:
        best_start, best_end = current_start, current_end
    
    # Extract the best region
    if best_end - best_start + 1 < 30:  # Minimum length
        return None
    
    trimmed_seq = record.seq[best_start:best_end + 1]
    trimmed_qual = qualities[best_start:best_end + 1]
    
    filtered_record = SeqRecord(
        seq=trimmed_seq,
        id=record.id,
        description=f"{record.description} filtered:{best_start}-{best_end + 1}"
    )
    filtered_record.letter_annotations['phred_quality'] = trimmed_qual
    
    return filtered_record


def calculate_sequence_stats(record: SeqRecord) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a DNA sequence.
    
    Args:
        record (SeqRecord): Input sequence record
        
    Returns:
        dict: Dictionary containing sequence statistics
    """
    seq_str = str(record.seq)
    stats = {
        'length': len(seq_str),
        'gc_content': round(GC(record.seq), 2),
        'base_counts': {
            'A': seq_str.count('A'),
            'T': seq_str.count('T'),
            'G': seq_str.count('G'),
            'C': seq_str.count('C'),
            'N': seq_str.count('N')
        }
    }
    
    # Add quality statistics if available
    if hasattr(record, 'letter_annotations') and 'phred_quality' in record.letter_annotations:
        qualities = record.letter_annotations['phred_quality']
        stats['quality_stats'] = {
            'mean_quality': round(statistics.mean(qualities), 2),
            'median_quality': round(statistics.median(qualities), 2),
            'min_quality': min(qualities),
            'max_quality': max(qualities),
            'bases_above_q20': sum(1 for q in qualities if q >= 20),
            'bases_above_q30': sum(1 for q in qualities if q >= 30)
        }
    
    return stats


def reverse_complement_sequence(record: SeqRecord) -> SeqRecord:
    """
    Generate reverse complement of a DNA sequence.
    
    Args:
        record (SeqRecord): Input sequence record
        
    Returns:
        SeqRecord: Reverse complement sequence record
    """
    rev_comp_seq = record.seq.reverse_complement()
    rev_comp_record = SeqRecord(
        seq=rev_comp_seq,
        id=f"{record.id}_revcomp",
        description=f"{record.description} reverse complement"
    )
    
    # Reverse quality scores if present
    if hasattr(record, 'letter_annotations') and 'phred_quality' in record.letter_annotations:
        rev_comp_record.letter_annotations['phred_quality'] = record.letter_annotations['phred_quality'][::-1]
    
    return rev_comp_record


def find_adapter_sequences(record: SeqRecord, adapter_seq: str = "AGATCGGAAGAG") -> Tuple[bool, int]:
    """
    Find adapter sequences in reads (simplified adapter detection).
    
    Args:
        record (SeqRecord): Input sequence record
        adapter_seq (str): Adapter sequence to search for
        
    Returns:
        tuple: (adapter_found, position) - position is -1 if not found
    """
    seq_str = str(record.seq)
    position = seq_str.find(adapter_seq)
    
    if position != -1:
        return True, position
    
    # Also check for partial matches at the end
    for i in range(len(adapter_seq) - 3, 0, -1):  # Check partial matches of 4+ bases
        partial_adapter = adapter_seq[:i]
        if seq_str.endswith(partial_adapter):
            return True, len(seq_str) - i
    
    return False, -1


def trim_adapter_sequences(record: SeqRecord, adapter_seq: str = "AGATCGGAAGAG") -> SeqRecord:
    """
    Remove adapter sequences from reads.
    
    Args:
        record (SeqRecord): Input sequence record
        adapter_seq (str): Adapter sequence to remove
        
    Returns:
        SeqRecord: Sequence with adapters removed
    """
    found, position = find_adapter_sequences(record, adapter_seq)
    
    if found and position > 30:  # Only trim if sufficient sequence remains
        trimmed_seq = record.seq[:position]
        trimmed_record = SeqRecord(
            seq=trimmed_seq,
            id=record.id,
            description=f"{record.description} adapter_trimmed"
        )
        
        # Trim quality scores if present
        if hasattr(record, 'letter_annotations') and 'phred_quality' in record.letter_annotations:
            trimmed_record.letter_annotations['phred_quality'] = record.letter_annotations['phred_quality'][:position]
        
        return trimmed_record
    
    return record


def process_single_end_reads(input_file: str, output_file: str, 
                           min_quality: int = 20, window_size: int = 10) -> Dict[str, int]:
    """
    Process single-end Illumina reads with quality control and trimming.
    
    Args:
        input_file (str): Path to input FASTQ file
        output_file (str): Path to output FASTQ file
        min_quality (int): Minimum quality threshold
        window_size (int): Sliding window size
        
    Returns:
        dict: Processing statistics
    """
    stats = {
        'total_reads': 0,
        'quality_trimmed': 0,
        'sliding_window_filtered': 0,
        'adapter_trimmed': 0,
        'passed_filters': 0,
        'too_short': 0
    }
    
    with open_fastq_file(input_file) as infile, open(output_file, 'w') as outfile:
        for record in SeqIO.parse(infile, 'fastq'):
            stats['total_reads'] += 1
            
            # Step 1: Quality trimming
            trimmed_record = quality_trim_single_read(record, min_quality)
            if trimmed_record is None:
                stats['too_short'] += 1
                continue
            if len(trimmed_record.seq) < len(record.seq):
                stats['quality_trimmed'] += 1
            
            # Step 2: Sliding window filtering
            filtered_record = sliding_window_quality_filter(trimmed_record, window_size, min_quality)
            if filtered_record is None:
                stats['too_short'] += 1
                continue
            if len(filtered_record.seq) < len(trimmed_record.seq):
                stats['sliding_window_filtered'] += 1
            
            # Step 3: Adapter trimming
            final_record = trim_adapter_sequences(filtered_record)
            if len(final_record.seq) < len(filtered_record.seq):
                stats['adapter_trimmed'] += 1
            
            # Write final record
            if len(final_record.seq) >= 30:  # Final length check
                SeqIO.write(final_record, outfile, 'fastq')
                stats['passed_filters'] += 1
            else:
                stats['too_short'] += 1
    
    return stats


def process_paired_end_reads(input_file_r1: str, input_file_r2: str,
                           output_file_r1: str, output_file_r2: str,
                           min_quality: int = 20, window_size: int = 10) -> Dict[str, int]:
    """
    Process paired-end Illumina reads with synchronized quality control.
    
    Args:
        input_file_r1 (str): Path to R1 FASTQ file
        input_file_r2 (str): Path to R2 FASTQ file
        output_file_r1 (str): Path to output R1 FASTQ file
        output_file_r2 (str): Path to output R2 FASTQ file
        min_quality (int): Minimum quality threshold
        window_size (int): Sliding window size
        
    Returns:
        dict: Processing statistics
    """
    stats = {
        'total_pairs': 0,
        'both_passed': 0,
        'r1_failed': 0,
        'r2_failed': 0,
        'both_failed': 0
    }
    
    with (open_fastq_file(input_file_r1) as r1_file,
          open_fastq_file(input_file_r2) as r2_file,
          open(output_file_r1, 'w') as out_r1,
          open(output_file_r2, 'w') as out_r2):
        
        r1_records = SeqIO.parse(r1_file, 'fastq')
        r2_records = SeqIO.parse(r2_file, 'fastq')
        
        for r1_record, r2_record in zip(r1_records, r2_records):
            stats['total_pairs'] += 1
            
            # Process R1
            r1_trimmed = quality_trim_single_read(r1_record, min_quality)
            if r1_trimmed:
                r1_filtered = sliding_window_quality_filter(r1_trimmed, window_size, min_quality)
                if r1_filtered:
                    r1_final = trim_adapter_sequences(r1_filtered)
                    r1_passed = len(r1_final.seq) >= 30
                else:
                    r1_passed = False
            else:
                r1_passed = False
            
            # Process R2
            r2_trimmed = quality_trim_single_read(r2_record, min_quality)
            if r2_trimmed:
                r2_filtered = sliding_window_quality_filter(r2_trimmed, window_size, min_quality)
                if r2_filtered:
                    r2_final = trim_adapter_sequences(r2_filtered)
                    r2_passed = len(r2_final.seq) >= 30
                else:
                    r2_passed = False
            else:
                r2_passed = False
            
            # Only keep pairs where both reads pass
            if r1_passed and r2_passed:
                SeqIO.write(r1_final, out_r1, 'fastq')
                SeqIO.write(r2_final, out_r2, 'fastq')
                stats['both_passed'] += 1
            elif r1_passed and not r2_passed:
                stats['r2_failed'] += 1
            elif not r1_passed and r2_passed:
                stats['r1_failed'] += 1
            else:
                stats['both_failed'] += 1
    
    return stats


def analyze_fastq_file(input_file: str) -> Dict[str, Any]:
    """
    Perform comprehensive analysis of a FASTQ file.
    
    Args:
        input_file (str): Path to FASTQ file
        
    Returns:
        dict: Comprehensive analysis results
    """
    analysis = {
        'total_sequences': 0,
        'total_bases': 0,
        'length_distribution': {},
        'quality_distribution': {},
        'gc_content_stats': [],
        'base_composition': {'A': 0, 'T': 0, 'G': 0, 'C': 0, 'N': 0}
    }
    
    with open_fastq_file(input_file) as infile:
        for record in SeqIO.parse(infile, 'fastq'):
            analysis['total_sequences'] += 1
            seq_len = len(record.seq)
            analysis['total_bases'] += seq_len
            
            # Length distribution
            analysis['length_distribution'][seq_len] = analysis['length_distribution'].get(seq_len, 0) + 1
            
            # GC content
            gc_content = GC(record.seq)
            analysis['gc_content_stats'].append(gc_content)
            
            # Base composition
            seq_str = str(record.seq)
            for base in 'ATGCN':
                analysis['base_composition'][base] += seq_str.count(base)
            
            # Quality distribution
            if hasattr(record, 'letter_annotations') and 'phred_quality' in record.letter_annotations:
                for qual in record.letter_annotations['phred_quality']:
                    analysis['quality_distribution'][qual] = analysis['quality_distribution'].get(qual, 0) + 1
    
    # Calculate summary statistics
    if analysis['gc_content_stats']:
        analysis['mean_gc_content'] = round(statistics.mean(analysis['gc_content_stats']), 2)
        analysis['median_gc_content'] = round(statistics.median(analysis['gc_content_stats']), 2)
    
    if analysis['length_distribution']:
        lengths = []
        for length, count in analysis['length_distribution'].items():
            lengths.extend([length] * count)
        analysis['mean_length'] = round(statistics.mean(lengths), 2)
        analysis['median_length'] = round(statistics.median(lengths), 2)
    
    return analysis


def main():
    """
    Main function demonstrating various DNA sequence processing functions.
    """
    parser = argparse.ArgumentParser(description='DNA Sequence Processing with Biopython')
    parser.add_argument('--mode', choices=['single', 'paired', 'analyze'], required=True,
                       help='Processing mode')
    parser.add_argument('--input1', required=True, help='Input FASTQ file (or R1 for paired-end)')
    parser.add_argument('--input2', help='R2 FASTQ file for paired-end mode')
    parser.add_argument('--output1', help='Output FASTQ file (or R1 for paired-end)')
    parser.add_argument('--output2', help='R2 output FASTQ file for paired-end mode')
    parser.add_argument('--min-quality', type=int, default=20, help='Minimum quality threshold')
    parser.add_argument('--window-size', type=int, default=10, help='Sliding window size')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        if not args.output1:
            print("Error: --output1 required for single-end mode")
            return
        
        print(f"Processing single-end reads: {args.input1}")
        stats = process_single_end_reads(args.input1, args.output1, 
                                       args.min_quality, args.window_size)
        
        print("\nProcessing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        if stats['total_reads'] > 0:
            pass_rate = (stats['passed_filters'] / stats['total_reads']) * 100
            print(f"  Pass rate: {pass_rate:.2f}%")
    
    elif args.mode == 'paired':
        if not all([args.input2, args.output1, args.output2]):
            print("Error: --input2, --output1, and --output2 required for paired-end mode")
            return
        
        print(f"Processing paired-end reads: {args.input1}, {args.input2}")
        stats = process_paired_end_reads(args.input1, args.input2, 
                                       args.output1, args.output2,
                                       args.min_quality, args.window_size)
        
        print("\nProcessing Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        if stats['total_pairs'] > 0:
            pass_rate = (stats['both_passed'] / stats['total_pairs']) * 100
            print(f"  Pass rate: {pass_rate:.2f}%")
    
    elif args.mode == 'analyze':
        print(f"Analyzing FASTQ file: {args.input1}")
        analysis = analyze_fastq_file(args.input1)
        
        print("\nFile Analysis:")
        print(f"  Total sequences: {analysis['total_sequences']}")
        print(f"  Total bases: {analysis['total_bases']}")
        print(f"  Mean length: {analysis.get('mean_length', 'N/A')}")
        print(f"  Median length: {analysis.get('median_length', 'N/A')}")
        print(f"  Mean GC content: {analysis.get('mean_gc_content', 'N/A')}%")
        print(f"  Median GC content: {analysis.get('median_gc_content', 'N/A')}%")
        
        print("\nBase composition:")
        total_bases = sum(analysis['base_composition'].values())
        for base, count in analysis['base_composition'].items():
            if total_bases > 0:
                percentage = (count / total_bases) * 100
                print(f"  {base}: {count} ({percentage:.2f}%)")


def demo_functions():
    """
    Demonstration function showing how to use individual functions.
    Run this to see examples of each function in action.
    """
    print("=== DNA Sequence Processing Demo ===\n")
    
    # Create a sample sequence record for demonstratio
