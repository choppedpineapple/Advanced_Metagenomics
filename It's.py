#!/usr/bin/env python3
import os
import time
import logging
import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='consensus_generation.log'
)
logger = logging.getLogger('consensus_generator')

def calculate_consensus_sequence(sequences):
    """
    Calculate a consensus sequence from a list of sequences.
    Uses a numpy-based approach for performance with large datasets.
    """
    if not sequences:
        return ""
    
    # Get the maximum length among all sequences
    max_length = max(len(seq) for seq in sequences)
    
    # Convert sequences to a matrix for faster processing
    # Create a 2D matrix initialized with 'N' characters
    seq_matrix = np.full((len(sequences), max_length), 'N', dtype='<U1')
    
    # Fill the matrix with sequence characters
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            seq_matrix[i, j] = base
    
    # Calculate consensus sequence
    consensus = []
    bases = ["A", "C", "G", "T", "N"]
    
    # For each position, find the most common base
    for j in range(max_length):
        # Count occurrences of each base at this position
        column = seq_matrix[:, j]
        base_counts = Counter(column)
        
        # Find the most common base, prioritizing ACTG over N
        most_common = None
        max_count = 0
        
        for base in bases:
            if base_counts.get(base, 0) > max_count:
                max_count = base_counts.get(base, 0)
                most_common = base
        
        consensus.append(most_common)
    
    return ''.join(consensus)

def process_file(input_file_path, output_queue=None):
    """
    Process a single input file to generate consensus sequences.
    """
    try:
        logger.info(f"Processing file: {input_file_path}")
        
        with open(input_file_path, 'r') as file_1:
            seq_list = []
            seq_count = []
            headers = []
            
            # Parse the file
            for line in file_1:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split(':')
                    if len(parts) < 2:
                        logger.warning(f"Malformed line in {input_file_path}: {line}")
                        continue
                    
                    seq = parts[0].strip()
                    header_info = parts[1]
                    
                    # Extract count from header - more safely
                    try:
                        header_parts = header_info.split('_')
                        count = int(header_parts[0])
                        seq_count.append(count)
                    except (ValueError, IndexError):
                        logger.warning(f"Could not parse count from header: {header_info}")
                        count = 1  # Default count if parsing fails
                        seq_count.append(count)
                    
                    # Extract max_key from header if available - with better error handling
                    try:
                        if 'maxkey' in header_info:
                            max_key = header_info.split('maxkey')[1].strip()
                        elif '_' in header_info and len(header_info.split('_')) > 1:
                            max_key = header_info.split('_')[1].strip()
                        else:
                            max_key = "unknown"  # Default if we can't parse it
                    except Exception:
                        max_key = "unknown"
                    
                    seq_list.append(seq)
                    headers.append((count, max_key))
                
                except Exception as e:
                    logger.error(f"Error processing line: {line}. Error: {e}")
                    continue
        
        # Skip processing if no valid sequences were found
        if not seq_list:
            logger.warning(f"No valid sequences found in {input_file_path}")
            return []
        
        # Calculate the total count across all sequences
        total_count = sum(seq_count)
        
        # Generate final header - safely
        file_base = os.path.basename(input_file_path)
        # Use the first header's max_key if available, otherwise use "unknown"
        max_key_value = headers[0][1] if headers and len(headers) > 0 else "unknown"
        final_header = f"{total_count}_{max_key_value}_file_{file_base}"
        
        # Calculate consensus sequence
        consensus_sequence = calculate_consensus_sequence(seq_list)
        
        # Format for output
        result = f">{final_header}\n{consensus_sequence}"
        
        logger.info(f"Processed {len(seq_list)} sequences in {input_file_path}")
        
        if output_queue is not None:
            output_queue.put(result)
            return None
        else:
            return [result]
    
    except Exception as e:
        logger.error(f"Error processing file {input_file_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser(description="Generate consensus sequences from multiple sequence files")
    parser.add_argument('input_dir', nargs='?', 
                        help='Directory containing input files')
    parser.add_argument('--output', default='consensus_file.txt',
                       help='Output file for consensus sequences')
    parser.add_argument('--threads', type=int, default=32,
                       help='Number of threads to use')
    parser.add_argument('--file-pattern', default='.txt',
                       help='File pattern to match (default: .txt)')
    args = parser.parse_args()
    
    # Handle the case where input_dir is provided as positional argument
    input_dir = args.input_dir
    if not input_dir:
        input_dir = '/mnt/storage-HDD05a/1.scrach/immuno-mahek/WS1-scrach/crp/result/'
    
    start_time = time.time()
    logger.info(f"Script started at: {time.ctime(start_time)}")
    
    # Find all input files
    file_pattern = args.file_pattern
    
    try:
        file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir) 
                    if f.endswith(file_pattern)]
        
        if not file_list:
            logger.error(f"No files found matching pattern '{file_pattern}' in {input_dir}")
            print(f"No files found matching pattern '{file_pattern}' in {input_dir}")
            return
        
        logger.info(f"Found {len(file_list)} files to process")
        print(f"Found {len(file_list)} files to process")
        
        # Process files in parallel
        results = []
        
        # Use ProcessPoolExecutor for CPU-bound tasks
        with ProcessPoolExecutor(max_workers=args.threads) as executor:
            # Map each file to a process
            try:
                # Process files in chunks to avoid memory issues
                chunk_size = 50  # Adjust based on file size and memory constraints
                for i in range(0, len(file_list), chunk_size):
                    chunk = file_list[i:i+chunk_size]
                    
                    # Map each file in the chunk to a process and collect results
                    chunk_results = list(tqdm(
                        executor.map(process_file, chunk),
                        total=len(chunk),
                        desc=f"Processing files ({i+1}-{min(i+chunk_size, len(file_list))})"
                    ))
                    
                    # Flatten results list (since each file returns a list)
                    for result_list in chunk_results:
                        if result_list:
                            results.extend(result_list)
                    
                    # Log progress
                    logger.info(f"Processed {min(i+chunk_size, len(file_list))} of {len(file_list)} files")
                    print(f"Processed {min(i+chunk_size, len(file_list))} of {len(file_list)} files")
            
            except Exception as e:
                logger.error(f"Error in parallel processing: {e}")
                print(f"Error in parallel processing: {e}")
        
        # Write all results to output file
        logger.info(f"Writing {len(results)} consensus sequences to {args.output}")
        print(f"Writing {len(results)} consensus sequences to {args.output}")
        with open(args.output, 'w') as out_file:
            for result in results:
                out_file.write(f"{result}\n")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}")
        print(f"Error: {e}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    logger.info(f"Script completed in {elapsed_time:.2f} seconds")
    print(f"Script started at: {time.ctime(start_time)}")
    print(f"Script ended at: {time.ctime(end_time)}")
    print(f"Total execution time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
