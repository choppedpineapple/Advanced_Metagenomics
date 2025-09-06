# Install required packages if not already installed
if (!require("qiime2R")) install.packages("devtools")
if (!require("qiime2R")) devtools::install_github("jbisanz/qiime2R")
if (!require("phyloseq")) install.packages("phyloseq")
if (!require("ggplot2")) install.packages("ggplot2")
if (!require("dplyr")) install.packages("dplyr")
if (!require("tidyr")) install.packages("tidyr")
if (!require("vegan")) install.packages("vegan")

# Load required libraries
library(qiime2R)
library(phyloseq)
library(ggplot2)
library(dplyr)
library(tidyr)
library(vegan)

# Import QIIME2 artifacts
# The qiime2R package simplifies importing QIIME2 artifacts into R [[13]]
feature_table <- read_qza("table.qza")
taxonomy <- read_qza("taxonomy.qza")

# Create phyloseq object
# You can manually build the phyloseq object from individual qza files [[20]]
ps <- phyloseq(
  otu_table(feature_table$data, taxa_are_rows = TRUE),
  tax_table(as.matrix(taxonomy$data))
)

# Quality Control and Filtering
# Remove taxa with no reads
ps <- prune_taxa(taxa_sums(ps) > 0, ps)

# Extract taxonomy table for filtering
tax_tab <- as.data.frame(tax_table(ps))

# Filter out unwanted taxa (mitochondria, chloroplast, eukaryota, NA)
# Taxonomy-based filtering removes unwanted lineages [[30]]
ps_filtered <- ps %>%
  tax_glom(taxrank = "Kingdom") %>%
  subset_taxa(!is.na(Kingdom)) %>%
  subset_taxa(!Kingdom %in% c("Eukaryota", "Archaea")) %>%
  subset_taxa(!Phylum %in% c("Mitochondria", "Chloroplast")) %>%
  tax_glom(taxrank = "Genus") %>%
  subset_taxa(!is.na(Genus)) %>%
  subset_taxa(!Genus %in% c("Mitochondria", "Chloroplast"))

# Remove taxa with zero counts after filtering
ps_filtered <- prune_taxa(taxa_sums(ps_filtered) > 0, ps_filtered)

# Aggregate rare taxa to reduce complexity in plots
# Aggregate rare taxa to show only abundant ones and group others as "Others" [[49]]
top_taxa <- names(sort(taxa_sums(ps_filtered), decreasing = TRUE)[1:20])
ps_agg <- ps_filtered
tax_agg <- as(tax_table(ps_agg), "matrix")
tax_agg[!rownames(tax_agg) %in% top_taxa, ] <- "Others"
tax_table(ps_agg) <- tax_table(tax_agg)
ps_agg <- merge_taxa(ps_agg, "Others")

# Alpha Diversity Analysis
# Calculate alpha diversity metrics
# Alpha diversity represents diversity within a sample [[28]]
alpha_metrics <- c("Observed", "Chao1", "ACE", "Shannon", "Simpson", "InvSimpson", "Fisher")
alpha_div <- estimate_richness(ps_agg, measures = alpha_metrics)

# Create alpha diversity plots
pdf("alpha_diversity_plots.pdf", width = 12, height = 8)
for(metric in alpha_metrics) {
  if(metric %in% colnames(alpha_div)) {
    p <- ggplot(alpha_div, aes_string(x = seq_len(nrow(alpha_div)), y = metric)) +
      geom_point(size = 3) +
      labs(title = paste("Alpha Diversity -", metric), 
           x = "Sample Index", 
           y = metric) +
      theme_minimal() +
      theme(axis.text.x = element_blank())
    print(p)
  }
}
dev.off()

# Beta Diversity Analysis
# Transform data for beta diversity (using relative abundances)
ps_rel <- transform_sample_counts(ps_agg, function(x) x / sum(x))

# Calculate distance matrices
# Beta diversity measures differences between communities [[22]]
dist_methods <- c("bray", "jaccard", "euclidean")
dist_list <- lapply(dist_methods, function(method) {
  distance(ps_rel, method = method)
})
names(dist_list) <- dist_methods

# Perform PCoA on distance matrices
pcoa_list <- lapply(dist_list, ordinate, method = "PCoA")

# Create beta diversity plots
pdf("beta_diversity_plots.pdf", width = 15, height = 5)
for(i in seq_along(pcoa_list)) {
  p <- plot_ordination(ps_rel, pcoa_list[[i]], color = "SampleID") +
    geom_point(size = 3) +
    labs(title = paste("Beta Diversity -", names(pcoa_list)[i])) +
    theme_minimal()
  print(p)
}
dev.off()

# Abundance Plots
# Create abundance bar plots with top taxa
# Show abundant taxa and group others as "Others" [[49]]
top_taxa_data <- ps_agg %>%
  tax_glom(taxrank = "Genus") %>%
  psmelt() %>%
  group_by(OTU) %>%
  summarise(total_abundance = sum(Abundance)) %>%
  arrange(desc(total_abundance)) %>%
  top_n(20, total_abundance) %>%
  pull(OTU)

# Melt phyloseq object for plotting
ps_melted <- psmelt(ps_agg) %>%
  mutate(OTU = ifelse(OTU %in% top_taxa_data, as.character(OTU), "Others"))

# Aggregate by sample and OTU
abundance_plot_data <- ps_melted %>%
  group_by(Sample, OTU) %>%
  summarize(Abundance = sum(Abundance), .groups = "drop")

# Create abundance bar plot
pdf("abundance_barplot.pdf", width = 12, height = 8)
abundance_plot <- ggplot(abundance_plot_data, aes(x = Sample, y = Abundance, fill = OTU)) +
  geom_bar(stat = "identity") +
  labs(title = "Taxonomic Composition", x = "Sample", y = "Relative Abundance") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_discrete(name = "Taxa")
print(abundance_plot)
dev.off()

# Heatmap of top taxa
# Create a heatmap of the most abundant taxa
top_taxa_matrix <- otu_table(ps_agg)[top_taxa[1:min(20, length(top_taxa))], ]
top_taxa_matrix <- as.matrix(top_taxa_matrix)

# Normalize for better visualization
top_taxa_matrix_norm <- top_taxa_matrix / rowSums(top_taxa_matrix)

pdf("top_taxa_heatmap.pdf", width = 12, height = 10)
heatmap_plot <- heatmap(top_taxa_matrix_norm, 
                        main = "Heatmap of Top Taxa",
                        xlab = "Samples", 
                        ylab = "Taxa")
dev.off()

# Prevalence plot
# Show prevalence of taxa across samples
prevalence_data <- ps_agg %>%
  psmelt() %>%
  group_by(OTU) %>%
  summarize(prevalence = sum(Abundance > 0) / length(Abundance),
            total_abundance = sum(Abundance),
            .groups = "drop") %>%
  arrange(desc(total_abundance)) %>%
  head(20)

pdf("taxa_prevalence_plot.pdf", width = 10, height = 8)
prevalence_plot <- ggplot(prevalence_data, aes(x = reorder(OTU, prevalence), y = prevalence)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Taxa Prevalence", x = "Taxa", y = "Prevalence (Fraction of samples)") +
  theme_minimal()
print(prevalence_plot)
dev.off()

# Summary statistics
cat("Analysis completed successfully!\n")
cat("Number of samples:", nsamples(ps_agg), "\n")
cat("Number of taxa after filtering:", ntaxa(ps_agg), "\n")
cat("Alpha diversity metrics calculated:", paste(alpha_metrics, collapse = ", "), "\n")
cat("Beta diversity distance methods used:", paste(dist_methods, collapse = ", "), "\n")

# Save the final phyloseq object
saveRDS(ps_agg, "filtered_phyloseq_object.rds")

cat("All plots saved as PDFs and filtered phyloseq object saved as RDS file.\n")
