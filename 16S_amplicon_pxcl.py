from Bio import SeqIO
import gzip

def quality_filter(record, min_qual=20, window_size=5):
    """Sliding window quality filter"""
    quals = record.letter_annotations["phred_quality"]
    for i in range(0, len(quals) - window_size + 1):
        window = quals[i:i+window_size]
        if sum(window)/window_size < min_qual:
            return record[:i]
    return record

def process_reads(r1_path, r2_path, output_dir, fwd_primer, rev_primer):
    with gzip.open(r1_path, "rt") as r1, \
         gzip.open(r2_path, "rt") as r2:
         
        records_r1 = (rec for rec in SeqIO.parse(r1, "fastq"))
        records_r2 = (rec for rec in SeqIO.parse(r2, "fastq"))
        
        filtered = []
        for r1_rec, r2_rec in zip(records_r1, records_r2):
            # Primer trimming
            if fwd_primer in r1_rec.seq and rev_primer in r2_rec.seq:
                r1_filtered = quality_filter(r1_rec)
                r2_filtered = quality_filter(r2_rec)
                filtered.append((r1_filtered, r2_filtered))
                
        # Write filtered reads
        SeqIO.write((r[0] for r in filtered), 
                   f"{output_dir}/filtered_R1.fastq.gz", "fastq")
        SeqIO.write((r[1] for r in filtered), 
                   f"{output_dir}/filtered_R2.fastq.gz", "fastq")

def merge_reads(r1, r2, min_overlap=20, max_mismatch=2):
    """Simple overlap-based merging"""
    overlap = min(len(r1), len(r2), min_overlap)
    for i in range(overlap, 0, -1):
        r1_end = r1.seq[-i:]
        r2_start = r2.seq[:i]
        
        mismatches = sum(1 for a,b in zip(r1_end, r2_start) if a != b)
        if mismatches <= max_mismatch:
            merged_seq = r1.seq + r2.seq[i:]
            merged_qual = r1.letter_annotations["phred_quality"] + \
                          r2.letter_annotations["phred_quality"][i:]
            return SeqRecord.SeqRecord(
                merged_seq,
                id=r1.id,
                description="",
                letter_annotations={"phred_quality": merged_qual}
            )
    return None  # No valid merge

import subprocess
import pandas as pd

def run_dada2(input_dir, output_path):
    """Wrapper for R DADA2 implementation"""
    cmd = [
        "Rscript",
        "dada2_script.R",
        "--input", input_dir,
        "--output", output_path
    ]
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print("DADA2 output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running DADA2:", e.stderr)
        raise

# Read ASV table
asv_table = pd.read_csv("asv_table.csv", index_col=0)

from collections import defaultdict

def build_kmer_db(ref_fasta, k=8):
    """Build k-mer index from reference sequences"""
    kmer_db = defaultdict(list)
    for record in SeqIO.parse(ref_fasta, "fasta"):
        seq = str(record.seq).upper()
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_db[kmer].append(record.id)
    return kmer_db

def assign_taxonomy(query_seq, kmer_db, k=8):
    """Simple taxonomy assignment"""
    matches = defaultdict(int)
    query_kmers = [query_seq[i:i+k] for i in range(len(query_seq)-k+1)]
    
    for kmer in query_kmers:
        for ref_id in kmer_db.get(kmer, []):
            matches[ref_id] += 1
            
    if matches:
        best_match = max(matches.items(), key=lambda x: x[1])
        return best_match[0]
    return "Unassigned"
import matplotlib.pyplot as plt
import seaborn as sns
from skbio.diversity import alpha_diversity

def plot_alpha_diversity(asv_table, metadata, metric='shannon'):
    # Calculate diversity
    counts = asv_table.values
    shannon = alpha_diversity(metric, counts)
    
    # Merge with metadata
    diversity_df = pd.DataFrame({
        'Sample': asv_table.index,
        metric: shannon
    }).merge(metadata, on='Sample')
    
    # Create plot
    plt.figure(figsize=(10,6))
    sns.boxplot(x='Treatment', y=metric, data=diversity_df)
    plt.title(f"{metric.capitalize()} Diversity by Treatment Group")
    plt.savefig("alpha_diversity.png")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory with raw FASTQs")
    parser.add_argument("metadata", help="Sample metadata file")
    parser.add_argument("-o", "--output", default="results", 
                       help="Output directory")
    args = parser.parse_args()
    
    # Create pipeline steps
    process_reads(args.input_dir, args.output)
    merge_reads(f"{args.output}/filtered_*")
    run_dada2(f"{args.output}/merged", f"{args.output}/asv_table.csv")
    # ... additional steps ...
    
    print(f"Pipeline complete. Results in {args.output}")

git init
git add scripts/ data/config/ requirements.txt
git commit -m "Initial pipeline implementation"
