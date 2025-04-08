#!/usr/bin/env python3

"""
S. aureus Primer Design Analysis
This script takes the k-mer candidates and evaluates them for use as PCR primers
"""

import subprocess
import os
import re
from Bio.Seq import Seq
from Bio import SeqIO

def calculate_tm(seq):
    """Calculate melting temperature using nearest-neighbor method"""
    # This is a simplified calculation - Primer3 is more accurate
    # You could call primer3_core here for better accuracy
    seq = seq.upper()
    A = seq.count('A')
    T = seq.count('T')
    G = seq.count('G')
    C = seq.count('C')
    
    # Simplified formula for sequences <14bp
    if len(seq) < 14:
        tm = (A + T) * 2 + (G + C) * 4
    else:
        # Wallace formula
        tm = 64.9 + 41 * (G + C - 16.4) / len(seq)
    
    return tm

def check_hairpin(seq):
    """Check for self-complementary regions that could form hairpins"""
    seq = seq.upper()
    rev_comp = str(Seq(seq).reverse_complement())
    
    for i in range(3, len(seq)-3):
        fragment = seq[i:i+4]
        if fragment in rev_comp:
            return True
    return False

def find_primer_pairs(kmers, distance_range=(100, 300)):
    """Find potential primer pairs that would amplify a region of appropriate size"""
    # In a real implementation, you would align the k-mers to the reference genome
    # and find pairs that are at an appropriate distance
    
    # This is a placeholder implementation
    potential_pairs = []
    
    # For a real implementation, you would do something like:
    # Map each k-mer to the reference genome
    # For k-mer A at position posA:
    #     For k-mer B at position posB:
    #         if (posB - posA) in distance_range:
    #             potential_pairs.append((A, B))
    
    return potential_pairs

def evaluate_kmers(kmer_file):
    """Evaluate k-mers for suitability as primers"""
    results = []
    
    with open(kmer_file, 'r') as f:
        kmers = [line.strip() for line in f if line.strip()]
    
    print(f"Evaluating {len(kmers)} potential primer sequences...")
    
    for kmer in kmers:
        tm = calculate_tm(kmer)
        gc_percent = (kmer.count('G') + kmer.count('C')) / len(kmer) * 100
        has_hairpin = check_hairpin(kmer)
        has_runs = bool(re.search(r'([ATGC])\1{3,}', kmer))  # Check for runs of 4+ same base
        
        # Compute a score (higher is better)
        score = 100
        
        # Ideal Tm is around 60°C
        tm_factor = abs(60 - tm)
        score -= tm_factor * 2
        
        # Penalize hairpins and runs
        if has_hairpin:
            score -= 30
        if has_runs:
            score -= 20
            
        # Ideal GC is 45-55%
        if gc_percent < 45 or gc_percent > 55:
            score -= abs(50 - gc_percent)
        
        results.append({
            'sequence': kmer,
            'tm': tm,
            'gc_percent': gc_percent,
            'has_hairpin': has_hairpin,
            'has_runs': has_runs,
            'score': score
        })
    
    # Sort by score (highest first)
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def main():
    print("S. aureus Primer Design Analysis")
    print("================================")
    
    # Adjust these paths to match your setup
    kmer_file = "kmer_results/primer_candidates.txt"
    
    # Evaluate k-mers
    results = evaluate_kmers(kmer_file)
    
    # Output top 10 candidates
    print("\nTop 10 Primer Candidates:")
    print("========================")
    print(f"{'Sequence':<25} {'Tm':<6} {'GC%':<6} {'Hairpin':<8} {'Runs':<6} {'Score':<6}")
    print("-" * 65)
    
    for i, r in enumerate(results[:10]):
        print(f"{r['sequence']:<25} {r['tm']:<6.1f} {r['gc_percent']:<6.1f} {r['has_hairpin']:<8} {r['has_runs']:<6} {r['score']:<6.1f}")
    
    # Generate a more comprehensive report
    with open("primer_evaluation_report.txt", "w") as f:
        f.write("S. aureus Primer Candidates Evaluation\n")
        f.write("====================================\n\n")
        
        for i, r in enumerate(results[:50]):
            f.write(f"Candidate #{i+1}: {r['sequence']}\n")
            f.write(f"  Melting Temperature: {r['tm']:.1f}°C\n")
            f.write(f"  GC Content: {r['gc_percent']:.1f}%\n")
            f.write(f"  Self-complementarity: {'Yes' if r['has_hairpin'] else 'No'}\n")
            f.write(f"  Nucleotide runs: {'Yes' if r['has_runs'] else 'No'}\n")
            f.write(f"  Score: {r['score']:.1f}\n\n")
    
    print(f"\nDetailed report written to primer_evaluation_report.txt")
    
    # For the best candidates, you would perform further analyses:
    # 1. BLAST against NCBI to confirm specificity
    # 2. Find complementary primer pairs
    # 3. Run in silico PCR to check for expected product sizes
    
    print("\nNext steps:")
    print("1. Validate top candidates using BLAST")
    print("2. Design complementary primers for PCR")
    print("3. Test in silico then in lab")

if __name__ == "__main__":
    main()
