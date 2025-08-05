import gzip
import sys
import math
from collections import Counter

def read_fastq(file_path):
    """Generator to read FASTQ records (handles gzipped files)"""
    open_func = gzip.open if file_path.endswith('.gz') else open
    with open_func(file_path, 'rt') as f:
        while True:
            header = f.readline().strip()
            if not header: break
            sequence = f.readline().strip()
            f.readline()  # Skip +
            quality = f.readline().strip()
            yield header[1:], sequence, quality  # Remove @ from header

def phred_to_prob(phred):
    """Convert Phred score to probability"""
    return 10 ** (-phred / 10)

def calculate_quality_stats(qualities):
    """Calculate mean and median quality scores"""
    scores = [ord(q) - 33 for q in qualities]
    mean = sum(scores) / len(scores)
    sorted_scores = sorted(scores)
    n = len(scores)
    median = (sorted_scores[n//2] + sorted_scores[(n-1)//2]) / 2
    return mean, median

def gc_content(sequence):
    """Calculate GC content percentage"""
    gc = sequence.count('G') + sequence.count('C')
    return (gc / len(sequence)) * 100

def process_fastq(input_file):
    """Main processing function"""
    read_count = 0
    total_bases = 0
    all_qualities = []
    gc_contents = []
    per_base_quality = Counter()
    base_counts = Counter()
    
    for header, sequence, quality in read_fastq(input_file):
        read_count += 1
        total_bases += len(sequence)
        gc_contents.append(gc_content(sequence))
        
        # Per-base quality tracking
        for i, q in enumerate(quality):
            per_base_quality[i] += ord(q) - 33
            base_counts[i] += 1
            
        # Store qualities for overall stats
        all_qualities.extend([ord(q) - 33 for q in quality])
        
        # Progress indicator
        if read_count % 10000 == 0:
            print(f"Processed {read_count} reads...")
    
    # Calculate final statistics
    avg_read_length = total_bases / read_count
    overall_gc = sum(gc_contents) / len(gc_contents)
    mean_qual, median_qual = calculate_quality_stats(all_qualities)
    
    # Per-base quality averages
    per_base_avg = {pos: per_base_quality[pos]/base_counts[pos] 
                   for pos in per_base_quality}
    
    return {
        'total_reads': read_count,
        'avg_read_length': avg_read_length,
        'overall_gc_content': overall_gc,
        'mean_quality': mean_qual,
        'median_quality': median_qual,
        'per_base_quality': per_base_avg
    }

def print_report(stats):
    """Print formatted analysis report"""
    print("\n=== FASTQ Quality Analysis Report ===")
    print(f"Total reads processed: {stats['total_reads']:,}")
    print(f"Average read length: {stats['avg_read_length']:.2f} bp")
    print(f"Overall GC content: {stats['overall_gc_content']:.2f}%")
    print(f"Mean quality score: {stats['mean_quality']:.2f}")
    print(f"Median quality score: {stats['median_quality']:.2f}")
    
    print("\nPer-base quality scores:")
    for pos, qual in sorted(stats['per_base_quality'].items()):
        print(f"  Base {pos+1:2d}: {qual:.2f}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fastq_pipeline.py <input.fastq.gz>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    stats = process_fastq(input_file)
    print_report(stats)
