#!/usr/bin/env python3
"""
Comprehensive Python for Bioinformatics Tutorial
================================================

This script demonstrates fundamental Python concepts with a focus on bioinformatics applications.
It covers functions, loops, conditionals, recursion, data structures, file operations,
and using key libraries like NumPy, Pandas, and BioPython.

Author: Claude
Date: March 7, 2025
"""

# =====================================================
# SECTION 1: IMPORTING PACKAGES AND BASIC SETUP
# =====================================================

# Standard library imports
import os                      # Operating system interface (file/directory operations)
import sys                     # System-specific parameters and functions
import re                      # Regular expressions for pattern matching
import time                    # Time access and conversions
import gzip                    # Work with gzipped files
import shutil                  # High-level file operations
import argparse                # Command-line argument parsing
from pathlib import Path       # Object-oriented filesystem paths
import multiprocessing as mp   # Parallel processing
from functools import partial  # Create partial functions with preset arguments
import random                  # Generate random numbers
import glob                    # File path pattern matching
import logging                 # Logging for debugging and monitoring

# Scientific computing imports
import numpy as np             # Numerical computing (arrays, math functions)
import pandas as pd            # Data manipulation and analysis
import matplotlib.pyplot as plt # Plotting and visualization

# Bioinformatics-specific imports
# Only import specific functions from Bio.SeqIO (more efficient)
from Bio.SeqIO import parse as seq_parse
from Bio.SeqIO import write as seq_write
from Bio.Seq import Seq        # Sequence object with biological methods
from Bio import SeqIO          # Sequence input/output
from Bio import AlignIO        # Sequence alignment input/output
from Bio.SeqRecord import SeqRecord  # Sequence with metadata

# Set up logging for better debugging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("bioinformatics_tutorial")

# =====================================================
# SECTION 2: BASIC PYTHON CONCEPTS
# =====================================================

# ---- Basic Variables and Data Types ----
def demonstrate_basic_variables():
    """
    Demonstrate basic Python variable types commonly used in bioinformatics.
    """
    # Integers - for counts, indices, etc.
    read_count = 1000
    
    # Floating point - for scores, p-values, etc.
    quality_score = 37.5
    
    # Strings - for sequences, IDs, etc.
    dna_sequence = "ATGCCGTAAGTC"
    gene_id = "ENSG00000139618"
    
    # Booleans - for conditions
    is_paired_end = True
    
    # Printing variables with formatted string (f-string)
    # This is the modern, preferred way to format strings
    print(f"Processing {read_count} reads with average quality score {quality_score}")
    print(f"Example sequence: {dna_sequence} (Gene ID: {gene_id})")
    print(f"Paired-end data: {is_paired_end}")
    
    # Type conversion
    # Convert string to integer
    coverage_str = "30"
    coverage_int = int(coverage_str)
    print(f"Coverage: {coverage_int} (converted from string '{coverage_str}')")
    
    # Convert integer to string
    read_length_int = 150
    read_length_str = str(read_length_int)
    print(f"Read length string: {read_length_str}")
    
    # Convert string to float
    gc_content_str = "42.3"
    gc_content_float = float(gc_content_str)
    print(f"GC content: {gc_content_float}%")

# ---- Lists, Tuples, Sets, and Dictionaries ----
def demonstrate_collections():
    """
    Demonstrate Python's collection data types and their bioinformatics applications.
    """
    # ---- Lists ----
    # Mutable ordered collections - can be modified after creation
    # Great for storing sequences of data that may change
    
    # Create a list of DNA sequences
    sequences = ["ATGC", "GGGCCC", "TTATATA", "GCGCGC"]
    print(f"Original sequences: {sequences}")
    
    # Accessing elements by index (0-based)
    first_seq = sequences[0]      # First element
    last_seq = sequences[-1]      # Last element
    print(f"First sequence: {first_seq}, Last sequence: {last_seq}")
    
    # Slicing lists [start:end:step] - end is exclusive
    subset = sequences[1:3]       # Elements at indices 1 and 2
    print(f"Subset sequences: {subset}")
    
    # Modifying lists
    sequences.append("AAAAAA")    # Add to the end
    print(f"After append: {sequences}")
    
    sequences.insert(1, "TTTT")   # Insert at specific position
    print(f"After insert: {sequences}")
    
    sequences.remove("GGGCCC")    # Remove by value
    print(f"After remove: {sequences}")
    
    popped_seq = sequences.pop()  # Remove and return last item
    print(f"Popped sequence: {popped_seq}")
    print(f"After pop: {sequences}")
    
    # List comprehensions - powerful way to create and transform lists
    lengths = [len(seq) for seq in sequences]
    print(f"Sequence lengths: {lengths}")
    
    # Filtered list comprehension
    long_seqs = [seq for seq in sequences if len(seq) > 4]
    print(f"Sequences longer than 4 nucleotides: {long_seqs}")
    
    # ---- Tuples ----
    # Immutable ordered collections - cannot be modified after creation
    # Good for fixed data or dictionary keys
    
    # Create a tuple of gene coordinates (chromosome, start, end)
    gene_loc = ("chr7", 140453136, 140481402)
    print(f"Gene location (tuple): {gene_loc}")
    
    # Unpacking tuple values into separate variables
    chromosome, start, end = gene_loc
    print(f"Chromosome: {chromosome}, Start: {start}, End: {end}")
    
    # Calculate length using unpacked values
    gene_length = end - start
    print(f"Gene length: {gene_length}bp")
    
    # ---- Sets ----
    # Unordered collections of unique elements
    # Useful for removing duplicates and set operations
    
    # Create sets of gene IDs from different analyses
    analysis1_genes = {"BRCA1", "TP53", "EGFR", "KRAS", "PTEN"}
    analysis2_genes = {"TP53", "KRAS", "MYC", "ALK", "BRAF"}
    
    # Set operations
    common_genes = analysis1_genes.intersection(analysis2_genes)
    print(f"Genes found in both analyses: {common_genes}")
    
    unique_to_analysis1 = analysis1_genes.difference(analysis2_genes)
    print(f"Genes unique to analysis 1: {unique_to_analysis1}")
    
    all_genes = analysis1_genes.union(analysis2_genes)
    print(f"All unique genes from both analyses: {all_genes}")
    
    # Remove duplicates from a list using a set
    reads_with_duplicates = ["read1", "read2", "read1", "read3", "read2"]
    unique_reads = list(set(reads_with_duplicates))
    print(f"Original reads: {reads_with_duplicates}")
    print(f"Unique reads: {unique_reads}")
    
    # ---- Dictionaries ----
    # Unordered collection of key-value pairs
    # Perfect for lookups and mappings
    
    # Create a dictionary of gene names and their functions
    gene_functions = {
        "BRCA1": "DNA repair",
        "TP53": "Tumor suppression",
        "KRAS": "Signal transduction",
        "EGFR": "Cell growth signaling"
    }
    print(f"Gene functions dictionary: {gene_functions}")
    
    # Accessing values by key
    tp53_function = gene_functions["TP53"]
    print(f"Function of TP53: {tp53_function}")
    
    # Using get() method (safer, returns None or default if key not found)
    mtor_function = gene_functions.get("MTOR", "Unknown function")
    print(f"Function of MTOR: {mtor_function}")
    
    # Adding or modifying entries
    gene_functions["MTOR"] = "Nutrient sensing"
    print(f"Updated dictionary: {gene_functions}")
    
    # Dictionary comprehension
    # Create a dictionary of sequence IDs and their GC content
    sequences = ["ATGC", "GGGCCC", "TTATATA", "GCGCGC"]
    
    def calculate_gc_content(seq):
        """Calculate GC content of a DNA sequence."""
        return ((seq.count('G') + seq.count('C')) / len(seq)) * 100
        
    gc_content = {f"seq{i}": calculate_gc_content(seq) 
                 for i, seq in enumerate(sequences)}
    print(f"GC content dictionary: {gc_content}")
    
    # Iterating through dictionaries
    print("\nGene functions:")
    for gene, function in gene_functions.items():
        print(f"{gene}: {function}")

# ---- Control Flow: Conditionals ----
def demonstrate_conditionals(sequence="ATGCCGTGA", quality_scores=[30, 35, 40, 20, 15, 25, 10, 40, 35]):
    """
    Demonstrate conditional statements (if/elif/else) in bioinformatics context.
    
    Args:
        sequence: A DNA sequence string
        quality_scores: List of Phred quality scores
    """
    # Basic if/else condition
    if len(sequence) % 3 == 0:
        print(f"Sequence length ({len(sequence)}) is divisible by 3 - could be a complete coding sequence")
    else:
        print(f"Sequence length ({len(sequence)}) is not divisible by 3 - possible incomplete coding sequence")
    
    # More complex if/elif/else condition
    # Check for start and stop codons
    start_codon = sequence[:3]
    stop_codon = sequence[-3:]
    
    if start_codon == "ATG" and stop_codon in ["TAA", "TAG", "TGA"]:
        print("Sequence has valid start and stop codons - complete gene")
    elif start_codon == "ATG":
        print("Sequence has valid start codon but no stop codon")
    elif stop_codon in ["TAA", "TAG", "TGA"]:
        print("Sequence has valid stop codon but no start codon")
    else:
        print("Sequence lacks both valid start and stop codons")
    
    # Nested conditions
    print("\nQuality assessment:")
    avg_quality = sum(quality_scores) / len(quality_scores)
    
    if avg_quality >= 30:
        print(f"High quality read (avg: {avg_quality:.1f})")
        
        # Nested condition
        min_quality = min(quality_scores)
        if min_quality < 20:
            print("  Warning: Some positions have low quality scores")
    elif avg_quality >= 20:
        print(f"Medium quality read (avg: {avg_quality:.1f})")
        
        # Check for stretches of low quality
        low_quality_stretch = False
        for i in range(len(quality_scores) - 2):
            if all(q < 20 for q in quality_scores[i:i+3]):
                low_quality_stretch = True
                break
                
        if low_quality_stretch:
            print("  Warning: Contains stretch of low quality bases")
    else:
        print(f"Low quality read (avg: {avg_quality:.1f}), consider discarding")
    
    # Conditional expressions (ternary operator)
    # Format: value_if_true if condition else value_if_false
    sequence_type = "protein-coding" if "ATG" in sequence else "non-coding"
    print(f"Sequence type: {sequence_type}")
    
    # Multiple conditions with logical operators (and, or, not)
    has_low_quality = any(score < 20 for score in quality_scores)
    has_high_gc = (sequence.count('G') + sequence.count('C')) / len(sequence) > 0.6
    
    if has_low_quality and has_high_gc:
        print("Sequence has both low quality regions and high GC content - may need special processing")
    elif has_low_quality or has_high_gc:
        print("Sequence has either low quality regions or high GC content - use caution")
    elif not has_low_quality and not has_high_gc:
        print("Sequence has neither low quality nor high GC content - good candidate for analysis")

# ---- Control Flow: Loops ----
def demonstrate_loops():
    """
    Demonstrate different types of loops in Python for bioinformatics applications.
    """
    print("\n----- FOR LOOPS -----")
    
    # Basic for loop with a list
    print("Basic for loop:")
    nucleotides = ['A', 'C', 'G', 'T']
    for nucleotide in nucleotides:
        print(f"Nucleotide: {nucleotide}")
    
    # For loop with range (for numeric iteration)
    print("\nFor loop with range:")
    for i in range(5):  # 0, 1, 2, 3, 4
        print(f"Index: {i}")
    
    # For loop with range (start, stop, step)
    print("\nFor loop with range (start, stop, step):")
    for i in range(10, 30, 5):  # 10, 15, 20, 25
        print(f"Position: {i}")
    
    # For loop with enumeration (tracking index)
    print("\nFor loop with enumeration:")
    bases = ['A', 'C', 'G', 'T']
    for index, base in enumerate(bases):
        print(f"Base at position {index}: {base}")
    
    # For loop through string (characters)
    print("\nFor loop through string:")
    dna_seq = "ATGC"
    for base in dna_seq:
        print(f"Base: {base}")
    
    # For loop through multiple lists simultaneously with zip
    print("\nFor loop with zip:")
    bases = ['A', 'G', 'C', 'T']
    counts = [354, 267, 289, 345]
    for base, count in zip(bases, counts):
        print(f"Count of {base}: {count}")
    
    # For loop with dictionary
    print("\nFor loop with dictionary:")
    codon_table = {'ATG': 'Methionine', 'TGG': 'Tryptophan', 'TAA': 'Stop'}
    for codon, amino_acid in codon_table.items():
        print(f"Codon {codon} codes for {amino_acid}")
    
    print("\n----- NESTED LOOPS -----")
    
    # Nested loops (loop within a loop)
    # Example: Calculating nucleotide transition frequencies
    print("Nucleotide transition frequencies:")
    
    nucleotides = ['A', 'C', 'G', 'T']
    for from_base in nucleotides:
        for to_base in nucleotides:
            if from_base != to_base:
                # In real data, you would count actual transitions
                # Here we just generate random frequencies
                frequency = round(random.random() * 0.1, 3)
                print(f"Transition {from_base}->{to_base}: {frequency}")
    
    # Another nested loop example: Processing paired-end reads
    print("\nProcessing paired-end reads:")
    read_pairs = [
        ('ACGT', 'TGCA'),
        ('GGTA', 'TACC'),
        ('AATT', 'TTAA')
    ]
    
    for i, (forward_read, reverse_read) in enumerate(read_pairs):
        print(f"Read pair {i+1}:")
        
        # Process each base in the forward read
        for j, base in enumerate(forward_read):
            # Get the corresponding base in the reverse read
            rev_base = reverse_read[j]
            print(f"  Position {j+1}: Forward={base}, Reverse={rev_base}")
    
    print("\n----- WHILE LOOPS -----")
    
    # While loop (continues until condition is no longer true)
    print("While loop example - sequence quality trimming:")
    
    # Simulate quality trimming of a read
    read = "ATGCTAGCTAGCTAGCATCG"
    quality_scores = [35, 34, 36, 37, 40, 38, 37, 35, 30, 25, 20, 15, 10, 8, 7, 6, 7, 6, 5, 4]
    
    trimmed_read = read
    min_quality = 20
    
    # Keep trimming from the end while quality is below threshold
    while quality_scores and quality_scores[-1] < min_quality:
        trimmed_read = trimmed_read[:-1]
        quality_scores.pop()
    
    print(f"Original read: {read} (length: {len(read)})")
    print(f"Trimmed read:  {trimmed_read} (length: {len(trimmed_read)})")
    print(f"Removed {len(read) - len(trimmed_read)} low-quality bases from the end")
    
    print("\n----- LOOP CONTROL -----")
    
    # Break statement (exit the loop completely)
    print("Break example - Find first stop codon:")
    
    sequence = "ATGCCGTAAGTCGTAGCTAGTGA"
    stop_codons = ["TAA", "TAG", "TGA"]
    
    for i in range(0, len(sequence)-2, 3):  # Step by 3 to check each codon
        codon = sequence[i:i+3]
        print(f"Checking codon: {codon}")
        
        if codon in stop_codons:
            print(f"Stop codon {codon} found at position {i}")
            break  # Exit the loop when first stop codon is found
    
    # Continue statement (skip the rest of the current iteration)
    print("\nContinue example - Process only A/T rich codons:")
    
    sequence = "ATGCCGTAGCTAGTTAA"
    
    for i in range(0, len(sequence)-2, 3):
        codon = sequence[i:i+3]
        
        # Skip codons with more G/C than A/T
        gc_count = codon.count('G') + codon.count('C')
        at_count = codon.count('A') + codon.count('T')
        
        if gc_count >= at_count:
            print(f"Skipping GC-rich codon: {codon}")
            continue
            
        print(f"Processing AT-rich codon: {codon}")

# ---- Functions ----
def demonstrate_functions():
    """
    Demonstrate creating and using functions in Python for bioinformatics applications.
    """
    print("\n----- BASIC FUNCTIONS -----")
    
    # Define a simple function to calculate GC content
    def calculate_gc_content(sequence):
        """
        Calculate the GC content of a DNA sequence.
        
        Args:
            sequence (str): DNA sequence
            
        Returns:
            float: GC content as a percentage
        """
        # Count G and C nucleotides
        gc_count = sequence.count('G') + sequence.count('C')
        total_length = len(sequence)
        
        # Avoid division by zero
        if total_length == 0:
            return 0
            
        # Calculate percentage
        gc_content = (gc_count / total_length) * 100
        return gc_content
    
    # Call the function with different sequences
    sequences = ["ATGCTA", "GCGCGC", "ATATAT"]
    
    for seq in sequences:
        gc = calculate_gc_content(seq)
        print(f"Sequence: {seq} | GC content: {gc:.1f}%")
    
    # Function with multiple parameters
    def calculate_nucleotide_frequency(sequence, nucleotide):
        """
        Calculate the frequency of a specific nucleotide in a DNA sequence.
        
        Args:
            sequence (str): DNA sequence
            nucleotide (str): Nucleotide to count (A, C, G, or T)
            
        Returns:
            float: Frequency as a percentage
        """
        # Count the specific nucleotide
        count = sequence.count(nucleotide.upper())
        total_length = len(sequence)
        
        # Avoid division by zero
        if total_length == 0:
            return 0
            
        # Calculate percentage
        frequency = (count / total_length) * 100
        return frequency
    
    # Test the function
    seq = "ATGCTACGTACGTAGC"
    for nucleotide in ['A', 'C', 'G', 'T']:
        freq = calculate_nucleotide_frequency(seq, nucleotide)
        print(f"Frequency of {nucleotide} in {seq}: {freq:.1f}%")
    
    print("\n----- FUNCTIONS WITH DEFAULT PARAMETERS -----")
    
    # Function with default parameters
    def translate_dna(sequence, start_pos=0, genetic_code=None):
        """
        Translate a DNA sequence to protein using the standard genetic code.
        
        Args:
            sequence (str): DNA sequence
            start_pos (int): Position to start translation (default: 0)
            genetic_code (dict): Custom genetic code (default: standard code)
            
        Returns:
            str: Protein sequence
        """
        # Standard genetic code (simplified)
        if genetic_code is None:
            genetic_code = {
                'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
                'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
                'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
                'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
                'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
                'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
                'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
                'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
                'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
                'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
                'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
                'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
                'T
