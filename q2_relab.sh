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


