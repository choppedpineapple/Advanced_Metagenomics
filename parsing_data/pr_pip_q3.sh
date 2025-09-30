#!/usr/bin/bash

# VDJ Heavy Chain Assembly Pipeline for Sheep Antibody Data
# Using pRESTO and IgBlast
# Input: ES-RAGE-heavy-S4_L001_R1_001.fastq.gz and ES-RAGE-heavy-S4_L001_R2_001.fastq.gz

set -e  # Exit on any error

# Configuration
THREADS=16
MEMORY=32G
INPUT_R1="ES-RAGE-heavy-S4_L001_R1_001.fastq.gz"
INPUT_R2="ES-RAGE-heavy-S4_L001_R2_001.fastq.gz"
SAMPLE_NAME="sheep_vh"

echo "Starting VDJ Heavy Chain Assembly Pipeline"
echo "Sample: $SAMPLE_NAME"
echo "Threads: $THREADS"
echo "Memory: $MEMORY"

# Create output directories
mkdir -p 01_quality_control 02_assembly 03_annotation 04_final

# Step 1: Quality control and filtering
echo "Step 1: Quality control and filtering with pRESTO"
ChangeQuality -s $INPUT_R1 -q 30 -t 4 --nproc $THREADS \
    | SplitFastq -n 1000000 -o 01_quality_control/filtered_R1_chunk_ --outDir ./
ChangeQuality -s $INPUT_R2 -q 30 -t 4 --nproc $THREADS \
    | SplitFastq -n 1000000 -o 01_quality_control/filtered_R2_chunk_ --outDir ./

# Merge paired-end reads using pRESTO
echo "Step 2: Merging paired-end reads"
for chunk in 01_quality_control/filtered_R1_chunk_*.fastq; do
    chunk_num=$(basename $chunk | sed 's/filtered_R1_chunk_\([0-9]*\).fastq/\1/')
    r2_chunk="01_quality_control/filtered_R2_chunk_${chunk_num}.fastq"
    
    if [ -f "$r2_chunk" ]; then
        echo "Processing chunk $chunk_num"
        # Filter sequences by length (expecting ~130bp fragments + overlap)
        FilterSeq -s $chunk -n -q 0.1 --outName 02_assembly/filtered_R1_${chunk_num} --nproc $((THREADS/2))
        FilterSeq -s $r2_chunk -n -q 0.1 --outName 02_assembly/filtered_R2_${chunk_num} --nproc $((THREADS/2))
        
        # Merge paired-end reads
        MakeOverlap -1 02_assembly/filtered_R1_${chunk_num}_filter.fastq \
                   -2 02_assembly/filtered_R2_${chunk_num}_filter.fastq \
                   -m 10 -x 0.3 --outName 02_assembly/merged_${chunk_num} --nproc $((THREADS/2))
    fi
done

# Combine all merged files
echo "Step 3: Combining merged files"
cat 02_assembly/merged_*_overlap.fastq > 02_assembly/combined_merged.fastq

# Step 4: Error correction and consensus calling
echo "Step 4: Error correction and consensus calling"
SplitFastq -s 02_assembly/combined_merged.fastq -n 500000 -o 02_assembly/consensus_input_chunk_ --outDir ./

for chunk in 02_assembly/consensus_input_chunk_*.fastq; do
    chunk_num=$(basename $chunk | sed 's/consensus_input_chunk_\([0-9]*\).fastq/\1/')
    BuildConsensus -s $chunk -n 0.5 --outName 02_assembly/consensus_${chunk_num} --nproc $((THREADS/2))
done

# Combine consensus sequences
cat 02_assembly/consensus_*_consensus.fastq > 02_assembly/final_consensus.fastq

# Clean up intermediate files
rm 02_assembly/consensus_input_chunk_*.fastq
rm 02_assembly/consensus_*_consensus.fastq

# Step 5: Quality filtering of consensus sequences
echo "Step 5: Final quality filtering"
FilterSeq -s 02_assembly/final_consensus.fastq -n -q 0.1 -l 100 --outName 02_assembly/vh_sequences --nproc $THREADS

# Step 6: IgBlast annotation
echo "Step 6: IgBlast annotation"
# Assuming IgBlast database for sheep is available
# You may need to adjust the database path and species accordingly
igblastn -query 02_assembly/vh_sequences_filter.fastq \
         -out 03_annotation/igblast_results.txt \
         -outfmt 19 \
         -organism sheep \
         -domain_system imgt \
         -ig_seqtype Ig \
         -show_translation \
         -num_threads $THREADS

# Step 7: Parse IgBlast results and extract successful annotations
echo "Step 7: Parsing IgBlast results"
python3 << 'EOF'
import pandas as pd
from Bio import SeqIO
import sys

def parse_igblast_results(igblast_file, input_fasta, output_fasta):
    # Read input sequences
    seq_dict = {rec.id: str(rec.seq) for rec in SeqIO.parse(input_fasta, "fasta")}
    
    # Parse IgBlast results
    results = []
    current_query = None
    current_v_gene = None
    current_j_gene = None
    current_productive = None
    
    with open(igblast_file, 'r') as f:
        for line in f:
            if line.startswith('# Query:'):
                if current_query is not None and current_productive:
                    results.append({
                        'query': current_query,
                        'v_gene': current_v_gene,
                        'j_gene': current_j_gene,
                        'productive': current_productive
                    })
                current_query = line.strip().split()[2]
                current_v_gene = None
                current_j_gene = None
                current_productive = False
            elif 'V gene' in line:
                current_v_gene = line.strip()
            elif 'J gene' in line:
                current_j_gene = line.strip()
            elif 'productively' in line:
                current_productive = True
    
    # Add last result
    if current_query is not None and current_productive:
        results.append({
            'query': current_query,
            'v_gene': current_v_gene,
            'j_gene': current_j_gene,
            'productive': current_productive
        })
    
    # Write successful annotations to output file
    with open(output_fasta, 'w') as out_f:
        for result in results:
            query_id = result['query']
            if query_id in seq_dict:
                header = f">{query_id}_V={result.get('v_gene', 'N/A')}_J={result.get('j_gene', 'N/A')}"
                out_f.write(f"{header}\n{seq_dict[query_id]}\n")

# Parse the results
parse_igblast_results('03_annotation/igblast_results.txt', 
                     '02_assembly/vh_sequences_filter.fastq', 
                     '04_final/annotated_vh_sequences.fasta')
EOF

# Alternative: Convert FASTQ to FASTA and create summary
echo "Step 8: Creating final output"
# Convert FASTQ to FASTA for IgBlast compatibility
awk 'BEGIN{RS="@"; FS="\n"} NR>1 {print ">"$1"\n"$2}' 02_assembly/vh_sequences_filter.fastq > 04_final/vh_sequences.fasta

# Create summary statistics
echo "Pipeline completed!" >> 04_final/pipeline_summary.txt
echo "Input sequences: $(zcat $INPUT_R1 | wc -l)" >> 04_final/pipeline_summary.txt
echo "Filtered sequences: $(grep -c '^@' 02_assembly/vh_sequences_filter.fastq)" >> 04_final/pipeline_summary.txt
echo "Successfully annotated sequences: $(grep -c '^>' 04_final/annotated_vh_sequences.fasta)" >> 04_final/pipeline_summary.txt

# Clean up temporary files if needed
# rm -rf 01_quality_control/filtered_R1_chunk_*.fastq 01_quality_control/filtered_R2_chunk_*.fastq
# rm -rf 02_assembly/filtered_R1_*.fastq 02_assembly/filtered_R2_*.fastq

echo "Pipeline completed successfully!"
echo "Final annotated sequences are in: 04_final/annotated_vh_sequences.fasta"
echo "IgBlast results are in: 03_annotation/igblast_results.txt"
echo "Summary statistics are in: 04_final/pipeline_summary.txt"
