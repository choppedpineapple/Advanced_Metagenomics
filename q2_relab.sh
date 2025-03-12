#!/usr/bin/bash

qiime feature-table relative-frequency \
  --i-table table.qza \
  --o-relative-frequency-table relative-table.qza

qiime feature-table summarize \
  --i-table relative-table.qza \
  --o-visualization relative-table-summary.qzv

#SampleID
Sample1
Sample2
Sample3
...

##### without metadata ####

#!/bin/bash

# Summarize feature table
qiime feature-table summarize \
  --i-table table.qza \
  --o-visualization table-summary.qzv

# Calculate relative frequencies
qiime feature-table relative-frequency \
  --i-table table.qza \
  --o-relative-frequency-table relative-table.qza

# Collapse feature table at genus level (level 6)
qiime taxa collapse \
  --i-table table.qza \
  --i-taxonomy taxonomy.qza \
  --p-level 6 \
  --o-collapsed-table table-genus.qza

# Filter features with total count less than 10
qiime feature-table filter-features \
  --i-table table.qza \
  --p-min-frequency 10 \
  --o-filtered-table filtered-table.qza

# Alpha rarefaction up to a depth of 10,000
qiime diversity alpha-rarefaction \
  --i-table table.qza \
  --p-max-depth 10000 \
  --o-visualization alpha-rarefaction.qzv

# Compute Shannon alpha diversity
qiime diversity alpha \
  --i-table table.qza \
  --p-metric shannon \
  --o-alpha-diversity shannon.qza

# Compute Bray-Curtis beta diversity
qiime diversity beta \
  --i-table table.qza \
  --p-metric braycurtis \
  --o-distance-matrix braycurtis.qza

# Export feature table to BIOM and TSV formats
qiime tools export \
  --input-path table.qza \
  --output-path exported-table

biom convert \
  --input-fp exported-table/feature-table.biom \
  --output-fp exported-table/feature-table.tsv \
  --to-tsv


  #### with fake metadata ####

# Create metadata.tsv from exported feature table
echo -e "#SampleID" > metadata.tsv
head -n1 exported-table/feature-table.tsv | cut -f2- | tr '\t' '\n' >> metadata.tsv

#SampleID
sample1
sample2
sample3
...

#!/bin/bash

# Taxonomic barplots
qiime taxa barplot \
  --i-table table.qza \
  --i-taxonomy taxonomy.qza \
  --m-metadata-file metadata.tsv \
  --o-visualization taxa-barplot.qzv

# Core diversity metrics (phylogenetic) with a sampling depth of 10,000
qiime diversity core-metrics-phylogenetic \
  --i-table table.qza \
  --i-phylogeny rooted-tree.qza \
  --p-sampling-depth 10000 \
  --m-metadata-file metadata.tsv \
  --output-dir core-metrics

# Emperor plot for unweighted UniFrac PCoA
qiime emperor plot \
  --i-pcoa core-metrics/unweighted_unifrac_pcoa_results.qza \
  --m-metadata-file metadata.tsv \
  --o-visualization unweighted-unifrac-pcoa.qzv

  

