#!/bin/bash
# Multi-language Metagenomics Analysis Pipeline
# This pipeline demonstrates using multiple languages for optimal performance
# in different computational tasks in metagenomics analysis

set -e  # Exit on error
set -u  # Treat unset variables as errors

# =========================================================
# Configuration and setup
# =========================================================

# Input parameters
INPUT_FASTQ="$1"  # Raw sequencing data
OUTPUT_DIR="$2"   # Output directory
THREADS="$3"      # Number of CPU threads to use

# Create output directories
mkdir -p "${OUTPUT_DIR}/qc"
mkdir -p "${OUTPUT_DIR}/assembly"
mkdir -p "${OUTPUT_DIR}/taxonomy"
mkdir -p "${OUTPUT_DIR}/functional"
mkdir -p "${OUTPUT_DIR}/visualization"

echo "Starting metagenomics analysis pipeline on $(date)"
echo "Input: ${INPUT_FASTQ}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Using ${THREADS} threads"

# =========================================================
# 1. Quality Control with Bash (Trimmomatic wrapper)
# =========================================================
# Bash is great for simple file operations and calling external tools

echo "[1/7] Running quality control with Trimmomatic..."
trimmomatic PE -threads "${THREADS}" \
    "${INPUT_FASTQ}_R1.fastq.gz" "${INPUT_FASTQ}_R2.fastq.gz" \
    "${OUTPUT_DIR}/qc/trimmed_R1_paired.fastq.gz" "${OUTPUT_DIR}/qc/trimmed_R1_unpaired.fastq.gz" \
    "${OUTPUT_DIR}/qc/trimmed_R2_paired.fastq.gz" "${OUTPUT_DIR}/qc/trimmed_R2_unpaired.fastq.gz" \
    ILLUMINACLIP:adapters.fa:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:36

# =========================================================
# 2. Custom sequence filtering with C++ (Much faster than Python/Bash)
# =========================================================
# C++ is excellent for compute-intensive operations like sequence processing

echo "[2/7] Running custom sequence filtering with C++..."
# Compile the C++ filtering program if needed
if [ ! -f "sequence_filter" ]; then
    echo "Compiling C++ sequence filter..."
    g++ -O3 -std=c++17 -o sequence_filter sequence_filter.cpp -lz
fi

# Run the C++ program for filtering
./sequence_filter \
    --input "${OUTPUT_DIR}/qc/trimmed_R1_paired.fastq.gz" \
    --input2 "${OUTPUT_DIR}/qc/trimmed_R2_paired.fastq.gz" \
    --output "${OUTPUT_DIR}/qc/filtered_R1.fastq.gz" \
    --output2 "${OUTPUT_DIR}/qc/filtered_R2.fastq.gz" \
    --min-quality 20 \
    --min-length 75 \
    --threads "${THREADS}"

# =========================================================
# 3. De novo assembly with Bash (MEGAHIT wrapper)
# =========================================================
# Bash works well for calling external tools with many parameters

echo "[3/7] Running de novo assembly with MEGAHIT..."
megahit -1 "${OUTPUT_DIR}/qc/filtered_R1.fastq.gz" \
        -2 "${OUTPUT_DIR}/qc/filtered_R2.fastq.gz" \
        -o "${OUTPUT_DIR}/assembly" \
        --min-contig-len 300 \
        --k-min 21 \
        --k-max 141 \
        --k-step 12 \
        --num-cpu-threads "${THREADS}"

# =========================================================
# 4. Custom taxonomic classification with Python (Using ML/AI models)
# =========================================================
# Python excels at machine learning and is good for this taxonomic classification task

echo "[4/7] Running taxonomic classification with Python ML model..."
python3 <<EOF
import os
import sys
import numpy as np
import pandas as pd
from Bio import SeqIO
import joblib
from sklearn.feature_extraction.text import CountVectorizer
import time

# Measure execution time
start_time = time.time()

# Load the pre-trained model
model = joblib.load("taxonomy_classifier.joblib")
vectorizer = joblib.load("kmer_vectorizer.joblib")
label_encoder = joblib.load("taxonomy_labels.joblib")

# Process contigs file
contigs_file = "${OUTPUT_DIR}/assembly/final.contigs.fa"
output_file = "${OUTPUT_DIR}/taxonomy/taxonomic_assignments.tsv"

# Parse contigs and extract k-mers
print("Processing contigs and extracting features...")
sequences = []
contig_ids = []

for record in SeqIO.parse(contigs_file, "fasta"):
    contig_ids.append(record.id)
    sequences.append(str(record.seq))

# Extract k-mer features
X = vectorizer.transform(sequences)

# Predict taxonomy
print("Predicting taxonomy...")
predictions = model.predict(X)
probabilities = model.predict_proba(X)
confidence = np.max(probabilities, axis=1)

# Get readable taxonomic labels
taxa = label_encoder.inverse_transform(predictions)

# Save results
result_df = pd.DataFrame({
    'contig_id': contig_ids,
    'predicted_taxon': taxa,
    'confidence': confidence
})

result_df.to_csv(output_file, sep='\t', index=False)

elapsed_time = time.time() - start_time
print(f"Taxonomic classification completed in {elapsed_time:.2f} seconds")
EOF

# =========================================================
# 5. Gene prediction with Prodigal (Bash wrapper)
# =========================================================
# Again, Bash is great for calling external tools

echo "[5/7] Predicting genes with Prodigal..."
prodigal -i "${OUTPUT_DIR}/assembly/final.contigs.fa" \
         -o "${OUTPUT_DIR}/functional/genes.gff" \
         -a "${OUTPUT_DIR}/functional/proteins.faa" \
         -d "${OUTPUT_DIR}/functional/genes.fna" \
         -p meta

# =========================================================
# 6. Functional annotation with parallel C++ tool
# =========================================================
# C++ is extremely efficient for the compute-intensive task of sequence alignment

echo "[6/7] Running functional annotation with C++..."
if [ ! -f "functional_annotator" ]; then
    echo "Compiling C++ functional annotator..."
    g++ -O3 -std=c++17 -fopenmp -o functional_annotator functional_annotator.cpp -lz
fi

./functional_annotator \
    --input "${OUTPUT_DIR}/functional/proteins.faa" \
    --db "protein_db.dmnd" \
    --output "${OUTPUT_DIR}/functional/annotations.tsv" \
    --threads "${THREADS}" \
    --evalue 1e-5 \
    --max-hits 50

# =========================================================
# 7. Statistical analysis and visualization with R
# =========================================================
# R excels at statistical analysis and visualization

echo "[7/7] Generating statistical analysis and visualizations with R..."
Rscript --vanilla <<EOF
library(tidyverse)
library(vegan)
library(phyloseq)
library(ggplot2)
library(viridis)

# Load taxonomic data
tax_data <- read_tsv("${OUTPUT_DIR}/taxonomy/taxonomic_assignments.tsv")

# Load functional data
func_data <- read_tsv("${OUTPUT_DIR}/functional/annotations.tsv")

# Count taxa at different levels
tax_counts <- tax_data %>%
  separate(predicted_taxon, c("Domain", "Phylum", "Class", "Order", "Family", "Genus", "Species"), 
           sep = ";", fill = "right") %>%
  group_by(Phylum) %>%
  summarize(count = n()) %>%
  filter(!is.na(Phylum)) %>%
  arrange(desc(count))

# Plot taxonomic composition
p1 <- ggplot(tax_counts %>% top_n(10, count), aes(x = reorder(Phylum, count), y = count, fill = Phylum)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  labs(x = "Phylum", y = "Count", title = "Top 10 Phyla in the Metagenome") +
  scale_fill_viridis(discrete = TRUE) +
  theme(legend.position = "none")

ggsave("${OUTPUT_DIR}/visualization/top_phyla.pdf", p1, width = 10, height = 6)

# Process functional annotations
func_summary <- func_data %>%
  group_by(function_category) %>%
  summarize(count = n()) %>%
  arrange(desc(count)) %>%
  top_n(20, count)

# Plot functional categories
p2 <- ggplot(func_summary, aes(x = reorder(function_category, count), y = count, fill = count)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  theme_minimal() +
  scale_fill_viridis() +
  labs(x = "Functional Category", y = "Count", title = "Top 20 Functional Categories") +
  theme(legend.position = "none")

ggsave("${OUTPUT_DIR}/visualization/top_functions.pdf", p2, width = 12, height = 8)

# Create a combined analysis report
write_lines(c(
  "# Metagenomics Analysis Report",
  paste("Analysis date:", Sys.Date()),
  "",
  "## Taxonomic Summary",
  paste("Total contigs analyzed:", nrow(tax_data)),
  paste("Number of phyla identified:", length(unique(tax_counts$Phylum))),
  "",
  "## Functional Summary",
  paste("Total genes predicted:", nrow(func_data)),
  paste("Number of functional categories:", length(unique(func_data$function_category)))
), "${OUTPUT_DIR}/visualization/report.md")

# Print completion message
cat("Statistical analysis and visualization completed\n")
EOF

# =========================================================
# Summarize runtime statistics
# =========================================================

echo "Pipeline completed successfully on $(date)"
echo "Output files are available in ${OUTPUT_DIR}"
echo ""
echo "Summary of output directories:"
echo "  - QC: ${OUTPUT_DIR}/qc"
echo "  - Assembly: ${OUTPUT_DIR}/assembly"
echo "  - Taxonomy: ${OUTPUT_DIR}/taxonomy"
echo "  - Functional: ${OUTPUT_DIR}/functional"
echo "  - Visualization: ${OUTPUT_DIR}/visualization"
