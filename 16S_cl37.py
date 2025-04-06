#!/usr/bin/env python3
"""
Quality Control Module

This module handles quality control and filtering of raw FASTQ files:
1. Primer trimming - removal of 16S PCR primer sequences
2. Quality filtering - removal of low-quality reads or bases
"""

import os
import gzip
import logging
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import re
from multiprocessing import Pool

logger = logging.getLogger('microbiome_pipeline.qc')

def detect_fastq_format(file_path):
    """
    Detect if a FASTQ file is gzipped based on file extension.
    
    Parameters:
    -----------
    file_path : str
        Path to the FASTQ file
    
    Returns:
    --------
    is_gzipped : bool
        True if the file is gzipped, False otherwise
    """
    return file_path.endswith('.gz')

def get_fastq_opener(file_path):
    """
    Get the appropriate opener function (open or gzip.open) based on file extension.
    
    Parameters:
    -----------
    file_path : str
        Path to the FASTQ file
    
    Returns:
    --------
    opener : function
        Function to open the file (either open or gzip.open)
    mode : str
        File opening mode ('rt' for text mode with gzip, 'r' for normal open)
    """
    if detect_fastq_format(file_path):
        return gzip.open, 'rt'
    else:
        return open, 'r'

def find_paired_reads(input_dir):
    """
    Find and pair forward and reverse read files in the input directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing FASTQ files
    
    Returns:
    --------
    paired_reads : list of tuples
        List of (sample_id, forward_read_path, reverse_read_path) tuples
    """
    # Common patterns for forward and reverse read file naming
    fwd_patterns = ['_R1', '_r1', '_1', '.R1', '.r1', '.1']
    rev_patterns = ['_R2', '_r2', '_2', '.R2', '.r2', '.2']
    
    # Collect all fastq files
    fastq_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(('.fastq', '.fq', '.fastq.gz', '.fq.gz')):
                fastq_files.append(os.path.join(root, file))
    
    # Group paired reads
    paired_reads = []
    processed_files = set()
    
    for file_path in fastq_files:
        if file_path in processed_files:
            continue
        
        file_name = os.path.basename(file_path)
        sample_id = None
        forward_path = None
        reverse_path = None
        
        # Check if this is a forward read
        for pattern in fwd_patterns:
            if pattern in file_name:
                # This is a forward read
                forward_path = file_path
                sample_id = file_name.split(pattern)[0]
                
                # Try to find the corresponding reverse read
                for rev_pattern in rev_patterns:
                    potential_rev_name = sample_id + rev_pattern + file_name.split(pattern)[1]
                    potential_rev_path = os.path.join(os.path.dirname(file_path), potential_rev_name)
                    
                    if os.path.exists(potential_rev_path):
                        reverse_path = potential_rev_path
                        break
                
                break
        
        # If we found a paired read
        if forward_path and reverse_path:
            paired_reads.append((sample_id, forward_path, reverse_path))
            processed_files.add(forward_path)
            processed_files.add(reverse_path)
    
    return paired_reads

def phred_score_to_quality(char, base=33):
    """
    Convert a Phred quality character to its numeric quality score.
    
    Parameters:
    -----------
    char : str
        A single character representing the Phred quality score
    base : int, optional
        Base value for Phred score encoding (33 for Phred+33, 64 for Phred+64)
    
    Returns:
    --------
    quality : int
        Numeric quality score
    """
    return ord(char) - base

def average_quality(quality_string, base=33):
    """
    Calculate the average quality score for a quality string.
    
    Parameters:
    -----------
    quality_string : str
        String of Phred quality scores
    base : int, optional
        Base value for Phred score encoding
    
    Returns:
    --------
    avg_quality : float
        Average quality score
    """
    if not quality_string:
        return 0
    
    total = sum(phred_score_to_quality(char, base) for char in quality_string)
    return total / len(quality_string)

def sliding_window_quality(quality_string, window_size=4, base=33):
    """
    Calculate average quality in a sliding window across the quality string.
    
    Parameters:
    -----------
    quality_string : str
        String of Phred quality scores
    window_size : int, optional
        Size of the sliding window
    base : int, optional
        Base value for Phred score encoding
    
    Returns:
    --------
    position : int
        Position where quality drops below threshold, or -1 if quality is good
    """
    if len(quality_string) < window_size:
        return average_quality(quality_string, base)
    
    scores = [phred_score_to_quality(char, base) for char in quality_string]
    
    # Calculate sliding window averages
    window_avgs = []
    for i in range(len(scores) - window_size + 1):
        window = scores[i:i+window_size]
        window_avgs.append(sum(window) / window_size)
    
    return window_avgs

def trim_primers(record, primer_fwd, primer_rev, max_mismatches=3):
    """
    Trim primer sequences from a sequence record.
    
    Parameters:
    -----------
    record : Bio.SeqRecord.SeqRecord
        Sequence record
    primer_fwd : str
        Forward primer sequence
    primer_rev : str
        Reverse primer sequence
    max_mismatches : int, optional
        Maximum number of mismatches allowed in primer sequence
    
    Returns:
    --------
    trimmed_record : Bio.SeqRecord.SeqRecord
        Sequence record with primers trimmed, or None if primers not found
    """
    seq_str = str(record.seq)
    
    # Try to find the forward primer
    fwd_match = re.search("(" + primer_fwd + "){e<=" + str(max_mismatches) + "}", seq_str)
    
    if fwd_match:
        start = fwd_match.end()
    else:
        # Forward primer not found
        return None
    
    # For the reverse read, we need to look for the reverse complement of the reverse primer
    primer_rev_rc = str(Seq(primer_rev).reverse_complement())
    rev_match = re.search("(" + primer_rev_rc + "){e<=" + str(max_mismatches) + "}", seq_str)
    
    if rev_match:
        end = rev_match.start()
    else:
        # Just trim the forward primer if reverse primer not found
        end = len(seq_str)
    
    # Create a new trimmed record
    if start >= end:
        # Invalid trimming positions
        return None
    
    trimmed_seq = record.seq[start:end]
    
    trimmed_record = SeqRecord(
        seq=trimmed_seq,
        id=record.id,
        name=record.name,
        description=record.description,
        letter_annotations={"phred_quality": record.letter_annotations["phred_quality"][start:end]}
    )
    
    return trimmed_record

def quality_filter(record, min_quality=20, window_size=4, min_length=100):
    """
    Filter a sequence record based on quality scores.
    
    Parameters:
    -----------
    record : Bio.SeqRecord.SeqRecord
        Sequence record
    min_quality : int, optional
        Minimum average quality score threshold
    window_size : int, optional
        Size of the sliding window for quality calculation
    min_length : int, optional
        Minimum acceptable length after trimming
    
    Returns:
    --------
    filtered_record : Bio.SeqRecord.SeqRecord
        Quality-filtered sequence record, or None if quality is too low
    """
    quality_string = record.letter_annotations["phred_quality"]
    
    # Calculate sliding window quality scores
    window_avgs = sliding_window_quality(quality_string, window_size)
    
    # Find the position where quality drops below threshold
    trim_pos = len(quality_string)
    for i, avg in enumerate(window_avgs):
        if avg < min_quality:
            trim_pos = i
            break
    
    # Trim the sequence at the position where quality drops
    if trim_pos < min_length:
        # Too short after trimming
        return None
    
    # Create a new trimmed record
    trimmed_seq = record.seq[:trim_pos]
    
    trimmed_record = SeqRecord(
        seq=trimmed_seq,
        id=record.id,
        name=record.name,
        description=record.description,
        letter_annotations={"phred_quality": record.letter_annotations["phred_quality"][:trim_pos]}
    )
    
    return trimmed_record

def process_paired_reads(args):
    """
    Process a single pair of paired-end reads.
    
    Parameters:
    -----------
    args : tuple
        Tuple containing (sample_id, forward_path, reverse_path, output_dir, parameters)
    
    Returns:
    --------
    stats : dict
        Statistics for the processed sample
    """
    sample_id, fwd_path, rev_path, output_dir, params = args
    
    # Extract parameters
    primer_fwd = params.get('primer_fwd')
    primer_rev = params.get('primer_rev')
    min_quality = params.get('min_quality', 20)
    window_size = params.get('window_size', 4)
    min_length = params.get('min_length', 100)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file paths
    out_fwd = os.path.join(output_dir, f"{sample_id}_R1_filtered.fastq")
    out_rev = os.path.join(output_dir, f"{sample_id}_R2_filtered.fastq")
    
    # Initialize counters
    stats = {
        'sample_id': sample_id,
        'input_reads': 0,
        'passed_reads': 0,
        'failed_primer': 0,
        'failed_quality': 0,
        'failed_length': 0
    }
    
    # Open input files
    fwd_opener, fwd_mode = get_fastq_opener(fwd_path)
    rev_opener, rev_mode = get_fastq_opener(rev_path)
    
    with fwd_opener(fwd_path, fwd_mode) as f_fwd, rev_opener(rev_path, rev_mode) as f_rev:
        # Parse FASTQ files
        fwd_records = SeqIO.parse(f_fwd, "fastq")
        rev_records = SeqIO.parse(f_rev, "fastq")
        
        # Open output files
        with open(out_fwd, 'w') as f_out_fwd, open(out_rev, 'w') as f_out_rev:
            # Process each pair of reads
            for fwd_record, rev_record in zip(fwd_records, rev_records):
                stats['input_reads'] += 1
                
                # Ensure reads are paired correctly (same ID)
                if fwd_record.id != rev_record.id:
                    logger.warning(f"Read pairing issue in sample {sample_id}: {fwd_record.id} vs {rev_record.id}")
                    continue
                
                # Trim primers if specified
                if primer_fwd and primer_rev:
                    fwd_trimmed = trim_primers(fwd_record, primer_fwd, primer_rev)
                    rev_trimmed = trim_primers(rev_record, primer_rev, primer_fwd)
                    
                    if not fwd_trimmed or not rev_trimmed:
                        stats['failed_primer'] += 1
                        continue
                else:
                    fwd_trimmed = fwd_record
                    rev_trimmed = rev_record
                
                # Quality filtering
                fwd_filtered = quality_filter(fwd_trimmed, min_quality, window_size, min_length)
                rev_filtered = quality_filter(rev_trimmed, min_quality, window_size, min_length)
                
                if not fwd_filtered or not rev_filtered:
                    if not fwd_filtered and not rev_filtered:
                        stats['failed_quality'] += 1
                    else:
                        stats['failed_length'] += 1
                    continue
                
                # Write filtered reads to output files
                SeqIO.write(fwd_filtered, f_out_fwd, "fastq")
                SeqIO.write(rev_filtered, f_out_rev, "fastq")
                
                stats['passed_reads'] += 1
    
    # Compress output files
    os.system(f"gzip {out_fwd}")
    os.system(f"gzip {out_rev}")
    
    return stats

def run_quality_control(config):
    """
    Run quality control and filtering on all samples.
    
    Parameters:
    -----------
    config : dict
        Pipeline configuration dictionary
    
    Returns:
    --------
    qc_stats : dict
        Quality control statistics for all samples
    """
    logger.info("Starting quality control and filtering")
    
    # Get input and output directories
    input_dir = config['input_dir']
    output_dir = os.path.join(config['output_dir'], 'processed_reads')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find paired-end reads
    paired_reads = find_paired_reads(input_dir)
    logger.info(f"Found {len(paired_reads)} paired read sets")
    
    if not paired_reads:
        logger.error("No paired reads found in the input directory")
        return {}
    
    # Extract QC parameters
    qc_params = {
        'primer_fwd': config.get('primer_fwd'),
        'primer_rev': config.get('primer_rev'),
        'min_quality': config.get('min_quality', 20),
        'window_size': config.get('window_size', 4),
        'min_length': config.get('min_length', 100)
    }
    
    # Process samples in parallel
    num_threads = config.get('threads', 4)
    logger.info(f"Processing {len(paired_reads)} samples using {num_threads} threads")
    
    # Prepare arguments for parallel processing
    process_args = [(sample_id, fwd, rev, output_dir, qc_params) 
                    for sample_id, fwd, rev in paired_reads]
    
    # Run processing in parallel
    with Pool(processes=num_threads) as pool:
        all_stats = pool.map(process_paired_reads, process_args)
    
    # Combine statistics
    qc_stats = {stats['sample_id']: stats for stats in all_stats}
    
    # Log summary statistics
    total_input = sum(stats['input_reads'] for stats in all_stats)
    total_passed = sum(stats['passed_reads'] for stats in all_stats)
    logger.info(f"QC Summary: {total_passed} / {total_input} reads passed ({total_passed/total_input*100:.2f}%)")
    
    # Write statistics to a file
    stats_file = os.path.join(config['output_dir'], 'qc_reports', 'qc_stats.tsv')
    os.makedirs(os.path.dirname(stats_file), exist_ok=True)
    
    with open(stats_file, 'w') as f:
        # Write header
        f.write("sample_id\tinput_reads\tpassed_reads\tfailed_primer\tfailed_quality\tfailed_length\tpassing_rate\n")
        
        # Write statistics for each sample
        for stats in all_stats:
            passing_rate = stats['passed_reads'] / stats['input_reads'] * 100 if stats['input_reads'] > 0 else 0
            f.write(f"{stats['sample_id']}\t{stats['input_reads']}\t{stats['passed_reads']}\t"
                   f"{stats['failed_primer']}\t{stats['failed_quality']}\t{stats['failed_length']}\t"
                   f"{passing_rate:.2f}%\n")
    
    logger.info(f"QC statistics written to {stats_file}")
    
    return qc_stats
