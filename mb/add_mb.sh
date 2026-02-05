#!/usr/bin/env bash

# ==========================================
# 0. SETUP & INPUTS
# ==========================================
# Activate QIIME2 environment
# source activate qiime2-2024.5

# DEFINE INPUTS HERE
TABLE="table.qza"
REP_SEQS="rep-seqs.qza"
TAXONOMY="taxonomy.qza"
# A tree if you did diversity analysis. If not, generate it or skip phylogenetic metrics.
TREE="rooted-tree.qza" 
METADATA="metadata.tsv" # <--- YOU MUST CREATE THIS

# OUTPUT DIRECTORY
OUT_DIR="tender_analysis_output"
mkdir -p $OUT_DIR

echo "Starting Analysis..."

# ==========================================
# 1. TAXONOMY & PERCENTAGE TABLES
# [cite_start]Tender Req: "Comprehensive tables showing percentage distribution of taxa per sample" [cite: 23]
# ==========================================
echo "Generating Percentage Tables..."

# Step 1A: Collapse ASVs to Genus level (Level 6)
# [cite_start]The tender explicitly asks for "minimum genus level" [cite: 23]
qiime taxa collapse \
  --i-table $TABLE \
  --i-taxonomy $TAXONOMY \
  --p-level 6 \
  --o-collapsed-table $OUT_DIR/table-l6-genus.qza

# Step 1B: Convert Counts to Relative Frequency (Percentages)
qiime feature-table relative-frequency \
  --i-table $OUT_DIR/table-l6-genus.qza \
  --o-relative-frequency-table $OUT_DIR/table-l6-genus-rel-freq.qza

# Step 1C: Export to CSV (The actual deliverable file)
# This creates a folder containing 'feature-table.biom'. We need to convert that to TSV.
qiime tools export \
  --input-path $OUT_DIR/table-l6-genus-rel-freq.qza \
  --output-path $OUT_DIR/exported-genus-table

# Convert BIOM to TSV (readable by Excel)
biom convert \
  -i $OUT_DIR/exported-genus-table/feature-table.biom \
  -o $OUT_DIR/FINAL_genus_percentage_table.tsv \
  --to-tsv

echo "Done. Deliverable located at: $OUT_DIR/FINAL_genus_percentage_table.tsv"

# ==========================================
# 2. DIVERSITY & STATS (With Metadata)
# [cite_start]Tender Req: "Comparative analysis between treatments and time points" [cite: 23]
# ==========================================
echo "Running Diversity Analysis..."

# This generates PCoA plots, Shannon, Bray-Curtis, etc.
# The metadata file allows you to visualize groups (Control vs Extract)
qiime diversity core-metrics-phylogenetic \
  --i-phylogeny $TREE \
  --i-table $TABLE \
  --p-sampling-depth 10000 \
  --m-metadata-file $METADATA \
  --output-dir $OUT_DIR/core-metrics

# ==========================================
# 3. DIFFERENTIAL ABUNDANCE (ANCOM-BC)
# [cite_start]Tender Req: "Optional statistical analysis of microbiome shifts" [cite: 23]
# ==========================================
echo "Running ANCOM-BC for statistical shifts..."

# Note: We use the Genus level table we created in Step 1A to make the results interpretable
qiime composition ancombc \
  --i-table $OUT_DIR/table-l6-genus.qza \
  --m-metadata-file $METADATA \
  --p-formula "treatment" \
  --o-differentials $OUT_DIR/ancombc-diffs.qza

# Export the statistical results to a readable table
qiime tools export \
  --input-path $OUT_DIR/ancombc-diffs.qza \
  --output-path $OUT_DIR/ancombc-results
# The result will be a TSV file inside that folder showing which Genera are significantly different.

# ==========================================
# 4. FUNCTIONAL ANNOTATION (PICRUSt2)
# [cite_start]Tender Req: "Functional annotation (e.g. KEGG, GO, eggNOG, Pfam)" [cite: 23]
# ==========================================
echo "Running PICRUSt2..."

# IMPORTANT: Likely need to switch environments here if you installed picrust2 separately.
# conda deactivate
# conda activate picrust2

# Run the full pipeline
# This predicts metagenomes (EC and KO numbers) from your 16S sequences
qiime picrust2 full-pipeline \
  --i-table $TABLE \
  --i-seqs $REP_SEQS \
  --output-dir $OUT_DIR/picrust2_output \
  --p-placement-tool epa-ng \
  --p-threads 24 \
  --p-hsp-method pic \
  --p-max-nsti 2 \
  --verbose

# Export the Pathway Abundance table (MetaCyc pathways)
qiime tools export \
  --input-path $OUT_DIR/picrust2_output/pathway_abundance.qza \
  --output-path $OUT_DIR/picrust2_exported

# Convert to TSV
biom convert \
  -i $OUT_DIR/picrust2_exported/feature-table.biom \
  -o $OUT_DIR/FINAL_functional_pathways.tsv \
  --to-tsv

echo "Analysis Complete. Check $OUT_DIR for all deliverables."
