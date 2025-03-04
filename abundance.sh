# Process all samples in parallel using GNU Parallel
mkdir -p bracken_output
find kraken2_reports/ -name "*.report" | parallel -j $CPU_THREADS \
  "bracken -d /storage/k2-db/ -i {} -o bracken_output/{/.}.bracken \
  -r 150 -l S -w bracken_output/{/.}_bracken_report.txt"

  combine_bracken_outputs.py --files bracken_output/*_species.bracken \
  -o combined_abundance.tsv

  # Convert Bracken reports to Krona format
mkdir -p krona_reports
for file in bracken_output/*_report.txt; do
  cut -f1,2 $file | sed 's/$/\t1/' > krona_reports/$(basename $file).krona
done

# Build interactive visualization
ktImportTaxonomy -o metagenome_profile.html krona_reports/*.krona



import pandas as pd
df = pd.read_csv('combined_abundance.tsv', sep='\t', index_col=0)
df = df.T.apply(lambda x: x/x.sum(), axis=1)  # Normalize to relative abundance
df.to_csv('metaphlan_input.tsv', sep='\t')

humann3 --input metaphlan_input.tsv \
        --output functional_profiles \
        --threads $CPU_THREADS \
        --pathways metacyc \
        --protein-database /storage/humann3/uniref

# In R
library(edgeR)
abundance_matrix <- read.delim("combined_abundance.tsv", row.names=1)
dge <- DGEList(counts=abundance_matrix)
dge <- estimateDisp(dge, design)
fit <- glmQLFit(dge, design)
qlf <- glmQLFTest(fit, contrast=contrasts)
topTags(qlf)


# Kraken2 confidence threshold (recommended >0.5)
kraken2 --db /storage/k2-db/ [...] --confidence 0.7

# Bracken minimum reads (species-level)
bracken [...] --threshold 1
