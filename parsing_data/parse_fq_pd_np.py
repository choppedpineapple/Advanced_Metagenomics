#!/usr/bin/env python3

import sys
import gzip
import pandas as pd
import numpy as np
from Bio import SeqIO
from pathlib import Path

def parse_fastq_file(filename):
    """
    Parse FASTQ file (regular or gzipped) and extract sequences, qualities, headers, and lengths.
    
    Args:
        filename (str): Path to FASTQ file
        
    Returns:
        tuple: (sequences, qualities, headers, lengths) as lists
    """
    sequences = []
    qualities = []
    headers = []
    lengths = []
    
    # Check if file is gzipped
    if filename.endswith('.gz'):
        handle = gzip.open(filename, 'rt')
    else:
        handle = open(filename, 'r')
    
    try:
        # Parse FASTQ using BioPython
        for record in SeqIO.parse(handle, "fastq"):
            sequences.append(str(record.seq))
            qualities.append(record.letter_annotations["phred_quality"])
            headers.append(record.description)
            lengths.append(len(record.seq))
    finally:
        handle.close()
    
    return sequences, qualities, headers, lengths

def create_pandas_dataframe(sequences, qualities, headers, lengths):
    """
    Create a pandas DataFrame from the parsed FASTQ data.
    
    Args:
        sequences (list): List of DNA sequences
        qualities (list): List of quality scores
        headers (list): List of sequence headers
        lengths (list): List of sequence lengths
        
    Returns:
        pd.DataFrame: DataFrame with all the data
    """
    df = pd.DataFrame({
        'header': headers,
        'sequence': sequences,
        'quality': qualities,
        'length': lengths
    })
    
    return df

def create_numpy_arrays(sequences, qualities, headers, lengths):
    """
    Create numpy arrays from the parsed FASTQ data.
    
    Args:
        sequences (list): List of DNA sequences
        qualities (list): List of quality scores
        headers (list): List of sequence headers
        lengths (list): List of sequence lengths
        
    Returns:
        dict: Dictionary containing numpy arrays
    """
    # Convert to numpy arrays
    # Note: sequences and headers are object arrays due to variable string lengths
    # qualities is a list of lists, so we keep it as object array
    numpy_data = {
        'headers': np.array(headers, dtype=object),
        'sequences': np.array(sequences, dtype=object),
        'qualities': np.array(qualities, dtype=object),
        'lengths': np.array(lengths, dtype=int)
    }
    
    return numpy_data

def main():
    """
    Main function to handle command line arguments and process FASTQ file.
    """
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python fastq_parser.py <fastq_file>")
        print("Example: python fastq_parser.py sample.fastq")
        print("Example: python fastq_parser.py sample.fastq.gz")
        sys.exit(1)
    
    fastq_file = sys.argv[1]
    
    # Check if file exists
    if not Path(fastq_file).exists():
        print(f"Error: File '{fastq_file}' not found!")
        sys.exit(1)
    
    try:
        print(f"Processing FASTQ file: {fastq_file}")
        
        # Parse FASTQ file
        sequences, qualities, headers, lengths = parse_fastq_file(fastq_file)
        
        print(f"Parsed {len(sequences)} sequences")
        
        # Create pandas DataFrame
        df = create_pandas_dataframe(sequences, qualities, headers, lengths)
        
        # Create numpy arrays
        numpy_data = create_numpy_arrays(sequences, qualities, headers, lengths)
        
        # Display results
        print("\n" + "="*50)
        print("PANDAS DATAFRAME:")
        print("="*50)
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst 5 rows:")
        print(df.head())
        
        print(f"\nData types:")
        print(df.dtypes)
        
        print(f"\nBasic statistics for sequence lengths:")
        print(df['length'].describe())
        
        print("\n" + "="*50)
        print("NUMPY ARRAYS:")
        print("="*50)
        print(f"Headers array shape: {numpy_data['headers'].shape}")
        print(f"Sequences array shape: {numpy_data['sequences'].shape}")
        print(f"Qualities array shape: {numpy_data['qualities'].shape}")
        print(f"Lengths array shape: {numpy_data['lengths'].shape}")
        
        print(f"\nFirst 3 headers:")
        for i, header in enumerate(numpy_data['headers'][:3]):
            print(f"  {i+1}: {header}")
        
        print(f"\nFirst 3 sequence lengths: {numpy_data['lengths'][:3]}")
        print(f"Length statistics - Min: {numpy_data['lengths'].min()}, Max: {numpy_data['lengths'].max()}, Mean: {numpy_data['lengths'].mean():.2f}")
        
        # Optional: Save to files
        output_base = Path(fastq_file).stem.replace('.fastq', '')
        
        # Save pandas DataFrame
        df.to_csv(f"{output_base}_pandas.csv", index=False)
        print(f"\nPandas DataFrame saved to: {output_base}_pandas.csv")
        
        # Save numpy arrays
        np.savez(f"{output_base}_numpy.npz", 
                headers=numpy_data['headers'],
                sequences=numpy_data['sequences'],
                qualities=numpy_data['qualities'],
                lengths=numpy_data['lengths'])
        print(f"NumPy arrays saved to: {output_base}_numpy.npz")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
