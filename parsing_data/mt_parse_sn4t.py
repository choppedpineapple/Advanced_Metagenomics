#!/usr/bin/env python3
"""
Multi-threaded FASTQ parser for Python 3.14 free-threaded build
Processes FASTQ files (plain or gzipped) using true parallelism
"""

import sys
import gzip
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, Tuple, List


def open_fastq(filepath: str):
    """Open FASTQ file, handling both plain and gzipped formats"""
    if filepath.endswith('.gz'):
        return gzip.open(filepath, 'rt')
    return open(filepath, 'r')


def parse_fastq_record(lines: List[str]) -> Tuple[str, str, str, str]:
    """
    Parse a single FASTQ record (4 lines)
    Returns: (header, sequence, plus_line, quality)
    """
    if len(lines) != 4:
        raise ValueError(f"Invalid FASTQ record: expected 4 lines, got {len(lines)}")
    
    header = lines[0].rstrip('
')
    sequence = lines[1].rstrip('
')
    plus_line = lines[2].rstrip('
')
    quality = lines[3].rstrip('
')
    
    # Basic validation
    if not header.startswith('@'):
        raise ValueError(f"Invalid FASTQ header: {header}")
    if not plus_line.startswith('+'):
        raise ValueError(f"Invalid plus line: {plus_line}")
    if len(sequence) != len(quality):
        raise ValueError(f"Sequence and quality lengths differ: {len(sequence)} vs {len(quality)}")
    
    return header, sequence, plus_line, quality


def read_fastq_chunks(filepath: str, chunk_size: int = 10000) -> Iterator[List[List[str]]]:
    """
    Read FASTQ file in chunks for parallel processing
    Each chunk contains multiple FASTQ records (4 lines each)
    """
    chunk = []
    record_lines = []
    
    with open_fastq(filepath) as f:
        for line in f:
            record_lines.append(line)
            
            # Complete FASTQ record has 4 lines
            if len(record_lines) == 4:
                chunk.append(record_lines)
                record_lines = []
                
                # Yield chunk when it reaches chunk_size
                if len(chunk) >= chunk_size:
                    yield chunk
                    chunk = []
        
        # Yield remaining records
        if chunk:
            yield chunk
        
        # Check for incomplete record
        if record_lines:
            print(f"Warning: Incomplete FASTQ record at end of file ({len(record_lines)} lines)",
                  file=sys.stderr)


def process_chunk(chunk_data: Tuple[int, List[List[str]]]) -> Tuple[int, List[dict]]:
    """
    Process a chunk of FASTQ records in parallel
    Returns: (chunk_id, list of processed records)
    """
    chunk_id, chunk = chunk_data
    processed_records = []
    
    for record_lines in chunk:
        try:
            header, sequence, plus_line, quality = parse_fastq_record(record_lines)
            
            # Example processing: calculate quality stats
            avg_quality = sum(ord(q) - 33 for q in quality) / len(quality)
            gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
            
            record_dict = {
                'read_id': header[1:].split()[0],  # Remove @ and get ID
                'sequence': sequence,
                'quality': quality,
                'length': len(sequence),
                'avg_quality': avg_quality,
                'gc_content': gc_content
            }
            processed_records.append(record_dict)
            
        except ValueError as e:
            print(f"Error in chunk {chunk_id}: {e}", file=sys.stderr)
            continue
    
    return chunk_id, processed_records


def parse_fastq_multithreaded(filepath: str, num_threads: int = 4, chunk_size: int = 10000):
    """
    Parse FASTQ file using multi-threading with Python 3.14 free-threading
    
    Args:
        filepath: Path to FASTQ file (can be gzipped)
        num_threads: Number of threads to use (default: 4)
        chunk_size: Number of records per chunk (default: 10000)
    """
    print(f"Processing {filepath} with {num_threads} threads...")
    print(f"Chunk size: {chunk_size} records")
    
    total_reads = 0
    total_bases = 0
    results_dict = {}
    
    # Create thread pool and submit chunks
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all chunks to the executor
        futures = {}
        for chunk_id, chunk in enumerate(read_fastq_chunks(filepath, chunk_size)):
            future = executor.submit(process_chunk, (chunk_id, chunk))
            futures[future] = chunk_id
        
        # Process results as they complete
        for future in as_completed(futures):
            chunk_id = futures[future]
            try:
                returned_chunk_id, processed_records = future.result()
                results_dict[returned_chunk_id] = processed_records
                
                total_reads += len(processed_records)
                total_bases += sum(r['length'] for r in processed_records)
                
                print(f"Chunk {returned_chunk_id}: {len(processed_records)} reads processed")
                
            except Exception as e:
                print(f"Error processing chunk {chunk_id}: {e}", file=sys.stderr)
    
    # Process results in order
    print("
=== Processing Complete ===")
    print(f"Total reads: {total_reads:,}")
    print(f"Total bases: {total_bases:,}")
    
    if total_reads > 0:
        print(f"Average read length: {total_bases / total_reads:.1f} bp")
        
        # Calculate overall statistics from all chunks
        all_records = []
        for chunk_id in sorted(results_dict.keys()):
            all_records.extend(results_dict[chunk_id])
        
        avg_quality_overall = sum(r['avg_quality'] for r in all_records) / len(all_records)
        avg_gc = sum(r['gc_content'] for r in all_records) / len(all_records)
        
        print(f"Average quality score: {avg_quality_overall:.2f}")
        print(f"Average GC content: {avg_gc:.2f}%")
        
        # Display first few records as examples
        print("
=== First 5 Records ===")
        for i, record in enumerate(all_records[:5]):
            print(f"
Read {i+1}:")
            print(f"  ID: {record['read_id']}")
            print(f"  Length: {record['length']} bp")
            print(f"  Avg Quality: {record['avg_quality']:.2f}")
            print(f"  GC Content: {record['gc_content']:.2f}%")
            print(f"  Sequence: {record['sequence'][:50]}...")
    
    return results_dict


def main():
    if len(sys.argv) != 2:
        print("Usage: python fastq_parser.py <fastq_file>", file=sys.stderr)
        print("  Supports both plain (.fastq) and gzipped (.fastq.gz) files", file=sys.stderr)
        sys.exit(1)
    
    fastq_file = sys.argv[1]
    
    # Check if file exists
    if not Path(fastq_file).exists():
        print(f"Error: File not found: {fastq_file}", file=sys.stderr)
        sys.exit(1)
    
    # Parse with 4 threads (user has 4 cores)
    results = parse_fastq_multithreaded(fastq_file, num_threads=4, chunk_size=10000)
    
    print("
=== Done ===")


if __name__ == "__main__":
    main()
