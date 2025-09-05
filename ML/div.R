# Load required libraries
library(phyloseq)
library(ggplot2)
library(dplyr)

# --- Assume 'physeq' is your phyloseq object created from table.qza and taxonomy.qza ---
# e.g., physeq <- qza_to_phyloseq(features = "table.qza", taxonomy = "taxonomy.qza")
print(physeq) # Check your object

# --- 1. Calculate and Plot Alpha Diversity ---
# Calculate Shannon diversity index
alpha_meas <- estimate_alpha_diversity(physeq, measure = "Shannon")

# Extract Sample IDs and Shannon values into a data frame for plotting
# phyloseq's estimate_alpha_diversity returns a vector named by Sample IDs
alpha_df <- data.frame(
  SampleID = names(alpha_meas),
  Shannon = unname(alpha_meas)
)

# Create a simple dot plot
alpha_plot <- ggplot(alpha_df, aes(x = SampleID, y = Shannon)) +
  geom_point(size = 3, color = "darkgreen") +
  theme_minimal() +
  labs(title = "Shannon Alpha Diversity", x = "Sample ID", y = "Shannon Index") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) # Rotate x-axis labels

# Display the plot
print(alpha_plot)

# --- 2. Calculate and Plot Beta Diversity ---
# Calculate Jaccard distance matrix
beta_dist <- distance(physeq, method = "jaccard")

# Perform Principal Coordinates Analysis (PCoA)
beta_pcoa <- ordinate(physeq, method = "PCoA", distance = beta_dist)

# Extract the eigenvalues to calculate % variance explained
# Eigenvalues are stored in the $values component for PCoA objects in phyloseq
# Note: Accessing eigenvalues can sometimes be object-specific.
# A common way is to check the structure or use scores()
eig <- beta_pcoa$values[, "Relative_eig"] # Try this first
if (is.null(eig)) {
  # If the above doesn't work, check the structure of beta_pcoa
  # str(beta_pcoa)
  # You might need to access them differently, e.g., beta_pcoa$CA$eig / sum(beta_pcoa$CA$eig) for CA/CCA
  # Or calculate from the scores if values aren't directly available
  warning("Eigenvalues not found in standard location. Percent variance might be NA.")
  var_explained_pc1 <- NA
  var_explained_pc2 <- NA
} else {
  var_explained_pc1 <- round(eig[1] * 100, 1)
  var_explained_pc2 <- round(eig[2] * 100, 1)
}

# Extract PCoA coordinates (scores) for plotting (usually PC1 and PC2)
# Use scores() function for robustness
pcoa_scores <- scores(beta_pcoa, display = "sites") # "sites" gets sample coordinates

# Ensure we have a data frame with at least two dimensions
if (ncol(pcoa_scores) >= 2) {
  # Create a data frame for ggplot
  beta_df <- data.frame(
    PC1 = pcoa_scores[, 1],
    PC2 = pcoa_scores[, 2],
    SampleID = rownames(pcoa_scores) # Sample IDs are usually row names
  )

  # Create axis labels with % variance explained
  x_label <- ifelse(is.na(var_explained_pc1), "PC1", paste0("PC1 (", var_explained_pc1, "%)"))
  y_label <- ifelse(is.na(var_explained_pc2), "PC2", paste0("PC2 (", var_explained_pc2, "%)"))

  # Create a simple scatter plot
  beta_plot <- ggplot(beta_df, aes(x = PC1, y = PC2)) +
    geom_point(size = 3, color = "purple") +
    theme_minimal() +
    labs(title = "PCoA - Jaccard Beta Diversity", x = x_label, y = y_label)

  # Display the plot
  print(beta_plot)

} else {
  message("Not enough dimensions in PCoA result to plot PC1 vs PC2.")
}

# --- Optional: Simple Taxa Bar Plot (as requested in previous steps) ---
# This uses the taxonomy.qza information already in 'physeq'
# Plot at Phylum level (taxonomic rank 2)
# Note: phyloseq might merge taxonomy below the specified level if not all levels are unique paths.
# Simple plot, might need filtering for readability if many taxa.
# Aggregate counts at Phylum level
physeq_phylum <- tax_glom(physeq, taxrank = "Phylum")

# Transform counts to relative abundance (proportions)
physeq_phylum_rel <- transform_sample_counts(physeq_phylum, function(x) x / sum(x))

# Melt the data for ggplot
# This requires converting to a standard format, phyloseq's plot_bar can help, or manual melt
# Using phyloseq's built-in plotting first (simpler)
# plot_bar(physeq_phylum_rel, fill = "Phylum") + facet_wrap(~SampleID, scales = "free_x")

# For a ggplot object you can customize:
# Extract the data
bar_plot_data <- psmelt(physeq_phylum_rel)

# Simple stacked bar plot
taxa_bar_plot <- ggplot(bar_plot_data, aes(x = Sample, y = Abundance, fill = Phylum)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Taxa Bar Plot (Phylum Level)", x = "Sample ID", y = "Relative Abundance")

print(taxa_bar_plot)

                                             
