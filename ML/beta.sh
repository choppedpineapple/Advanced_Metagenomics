qiime diversity core-metrics-phylogenetic \
  --i-table table.qza \
  --i-phylogeny rep-seqs.qza  \
  --p-sampling-depth 1000 \
  --m-metadata-file fake-metadata.tsv \
  --output-dir core-metrics-results


qiime composition add-pseudocount \
  --i-table table.qza \
  --o-composition-table comp-table.qza

qiime composition ancombc \
  --i-table comp-table.qza \
  --m-metadata-file fake-metadata.tsv \
  --p-formula group \
  --p-p-adj-method BH \
  --o-differentials ancombc-diff.qza

qiime composition da-barplot \
  --i-data ancombc-diff.qz
  --p-significance-threshold 0.05 \
  --o-visualization da-barplot.qzv
