#!/usr/bin/env Rscript

# =========================
# Non-phylogenetic diversity & abundance from QIIME2 artifacts
# Inputs : table.qza (feature table), taxonomy.qza (q2-feature-classifier output)
# Output : metrics (CSV/TSV) + plots (PNG/PDF) in output_dir
# Usage  : Rscript qiime2_alpha_beta_no_phylo.R table.qza taxonomy.qza output_dir
# =========================

# ---- Args ----
args <- commandArgs(trailingOnly = TRUE)
table_fp   <- ifelse(length(args) >= 1, args[1], "table.qza")
tax_fp     <- ifelse(length(args) >= 2, args[2], "taxonomy.qza")
outdir     <- ifelse(length(args) >= 3, args[3], "qiime2_r_results")

# ---- Packages (auto-install if missing) ----
req_pkgs <- c(
  "qiime2R","phyloseq","tidyverse","vegan","data.table","pheatmap","patchwork","RColorBrewer"
)
install_if_missing <- function(pkgs){
  for(p in pkgs){
    if(!requireNamespace(p, quietly = TRUE)){
      install.packages(p, repos = "https://cloud.r-project.org")
    }
    suppressPackageStartupMessages(library(p, character.only = TRUE))
  }
}
install_if_missing(req_pkgs)

# ---- Helpers ----
safe_mkdir <- function(d){ if(!dir.exists(d)) dir.create(d, recursive = TRUE) }
safe_mkdir(outdir)
fpath <- function(...) file.path(outdir, ...)

# ---- Import QIIME2 artifacts ----
# qza_to_phyloseq will create a phyloseq object with otu_table + tax_table
message("Reading QIIME2 artifacts...")
ps <- qiime2R::qza_to_phyloseq(
  features = table_fp,
  taxonomy = tax_fp
)

# Ensure sample_data exists (minimal metadata: sample_id + library_size)
if(is.null(phyloseq::sample_data(ps, errorIfNULL = FALSE))){
  samp_ids <- phyloseq::sample_names(ps)
  libsize  <- colSums(phyloseq::otu_table(ps))
  meta_df  <- data.frame(Sample = samp_ids, LibrarySize = libsize, row.names = samp_ids, check.names = FALSE)
  phyloseq::sample_data(ps) <- phyloseq::sample_data(meta_df)
}

# Replace missing taxonomy ranks with "Unassigned"
if(!is.null(phyloseq::tax_table(ps, errorIfNULL = FALSE))){
  tax <- as.data.frame(phyloseq::tax_table(ps))
  tax[is.na(tax) | tax == "" ] <- "Unassigned"
  phyloseq::tax_table(ps) <- as.matrix(tax)
}

# ---- Save basic summaries ----
message("Saving summaries...")
# Sample read counts
sample_counts <- data.frame(Sample = phyloseq::sample_names(ps),
                            LibrarySize = colSums(phyloseq::otu_table(ps)),
                            check.names = FALSE)
data.table::fwrite(sample_counts, fpath("sample_library_sizes.csv"))

# Feature prevalence
feat_prev <- data.frame(
  FeatureID = rownames(phyloseq::otu_table(ps)),
  Prevalence = rowSums(phyloseq::otu_table(ps) > 0),
  TotalAbundance = rowSums(phyloseq::otu_table(ps)),
  check.names = FALSE
)
data.table::fwrite(feat_prev, fpath("feature_prevalence.csv"))

# ---- Alpha diversity (non-phylogenetic) ----
message("Computing alpha diversity...")
alpha_df <- phyloseq::estimate_richness(ps, measures = c("Observed","Shannon","Simpson","InvSimpson"))
alpha_df$Evenness_Pielou <- with(alpha_df, ifelse(Observed > 0, Shannon / log(Observed), NA_real_))
alpha_df$Sample <- rownames(alpha_df)
alpha_df <- cbind(alpha_df, data.frame(phyloseq::sample_data(ps)[alpha_df$Sample, , drop = FALSE]))
data.table::fwrite(alpha_df, fpath("alpha_diversity.csv"))

# Alpha plots: richness & evenness vs library size
g_alpha1 <- ggplot(alpha_df, aes(LibrarySize, Observed)) +
  geom_point() + geom_smooth(method = "loess", se = TRUE) +
  labs(title = "Observed Features vs Library Size",
       x = "Library size (reads)", y = "Observed features")
g_alpha2 <- ggplot(alpha_df, aes(LibrarySize, Shannon)) +
  geom_point() + geom_smooth(method = "loess", se = TRUE) +
  labs(title = "Shannon Diversity vs Library Size",
       x = "Library size (reads)", y = "Shannon")
g_alpha3 <- ggplot(alpha_df, aes(LibrarySize, Evenness_Pielou)) +
  geom_point() + geom_smooth(method = "loess", se = TRUE) +
  labs(title = "Pielou's Evenness vs Library Size",
       x = "Library size (reads)", y = "Evenness")

ggsave(fpath("alpha_vs_depth.png"), (g_alpha1 / g_alpha2 / g_alpha3), width = 8, height = 12, dpi = 300)

# Rarefaction curves (vegan)
message("Drawing rarefaction curves...")
otu <- as.matrix(phyloseq::otu_table(ps))
png(fpath("rarefaction_curves.png"), width = 2000, height = 1600, res = 220)
vegan::rarecurve(t(otu), step = 100, cex = 0.6, label = TRUE)
dev.off()

# ---- Beta diversity (non-phylogenetic) ----
message("Computing beta diversity distances...")
dist_bray   <- phyloseq::distance(ps, method = "bray")
dist_jacc   <- phyloseq::distance(ps, method = "jaccard", binary = TRUE)

# Save distance matrices
data.table::fwrite(as.data.frame(as.matrix(dist_bray)), fpath("beta_bray_curtis.tsv"), sep = "\t")
data.table::fwrite(as.data.frame(as.matrix(dist_jacc)), fpath("beta_jaccard.tsv"), sep = "\t")

# Ordinations: PCoA on Bray and Jaccard
ord_bray <- phyloseq::ordinate(ps, method = "PCoA", distance = dist_bray)
ord_jacc <- phyloseq::ordinate(ps, method = "PCoA", distance = dist_jacc)

p_bray <- phyloseq::plot_ordination(ps, ord_bray, type = "samples") +
  geom_point(size = 3) + labs(title = "PCoA (Bray–Curtis)")
p_jacc <- phyloseq::plot_ordination(ps, ord_jacc, type = "samples") +
  geom_point(size = 3) + labs(title = "PCoA (Jaccard)")

ggsave(fpath("pcoa_bray.png"), p_bray, width = 7, height = 6, dpi = 300)
ggsave(fpath("pcoa_jaccard.png"), p_jacc, width = 7, height = 6, dpi = 300)

# Also NMDS (stress reported in plot subtitle)
set.seed(42)
ord_nmds_bray <- phyloseq::ordinate(ps, method = "NMDS", distance = dist_bray, trymax = 100)
stress_txt <- paste0("stress = ", round(ord_nmds_bray$stress, 3))
p_nmds <- phyloseq::plot_ordination(ps, ord_nmds_bray, type = "samples") +
  geom_point(size = 3) + labs(title = "NMDS (Bray–Curtis)", subtitle = stress_txt)
ggsave(fpath("nmds_bray.png"), p_nmds, width = 7, height = 6, dpi = 300)

# ---- Abundance: relative abundance & taxonomic summaries ----
message("Building abundance summaries...")
ps_rel <- phyloseq::transform_sample_counts(ps, function(x) x / sum(x))

# Function to aggregate and plot top-N taxa at a chosen rank
plot_topN_rank <- function(ps_obj, rank = "Phylum", topN = 15, file_prefix = "phylum"){
  if(is.null(phyloseq::tax_table(ps_obj, errorIfNULL = FALSE))){
    warning("No taxonomy available; skipping ", rank, " plot.")
    return(invisible(NULL))
  }
  # Agglomerate
  ps_glom <- tryCatch(phyloseq::tax_glom(ps_obj, taxrank = rank, NArm = FALSE),
                      error = function(e) { message("tax_glom failed: ", e$message); return(NULL) })
  if(is.null(ps_glom)) return(invisible(NULL))

  # Melt to long format
  df <- phyloseq::psmelt(ps_glom) %>%
    dplyr::mutate(Taxon = get(rank)) %>%
    dplyr::mutate(Taxon = ifelse(is.na(Taxon) | Taxon == "", "Unassigned", Taxon))

  # Pick topN by mean rel. abundance
  top_taxa <- df %>% dplyr::group_by(Taxon) %>%
    dplyr::summarise(MeanAbund = mean(Abundance, na.rm = TRUE)) %>%
    dplyr::arrange(desc(MeanAbund)) %>% dplyr::slice_head(n = topN) %>% dplyr::pull(Taxon)

  df$TaxonCollapsed <- ifelse(df$Taxon %in% top_taxa, as.character(df$Taxon), "Other")

  p <- ggplot(df, aes(x = Sample, y = Abundance, fill = TaxonCollapsed)) +
    geom_bar(stat = "identity") +
    labs(title = paste0("Relative Abundance @ ", rank, " (Top ", topN, ")"),
         y = "Relative abundance", x = "Sample") +
    theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

  ggsave(fpath(paste0(file_prefix, "_stacked_bar.png")), p, width = 12, height = 6, dpi = 300)

  # Save table of mean relative abundance by taxon
  mean_tbl <- df %>% dplyr::group_by(Taxon) %>%
    dplyr::summarise(MeanRelAbundance = mean(Abundance, na.rm = TRUE)) %>%
    dplyr::arrange(desc(MeanRelAbundance))
  data.table::fwrite(mean_tbl, fpath(paste0(file_prefix, "_mean_rel_abundance.csv")))
}

# Stacked bars at Phylum and Genus
plot_topN_rank(ps_rel, rank = "Phylum", topN = 15, file_prefix = "phylum")
plot_topN_rank(ps_rel, rank = "Genus",  topN = 20, file_prefix = "genus")

# Heatmap of top 50 features by mean relative abundance
message("Drawing heatmaps...")
rel_mat <- as.matrix(phyloseq::otu_table(ps_rel))
topN <- min(50, nrow(rel_mat))
top_idx <- order(rowMeans(rel_mat, na.rm = TRUE), decreasing = TRUE)[seq_len(topN)]
mat_top <- rel_mat[top_idx, , drop = FALSE]
# annotate with taxonomy if present
row_ann <- NULL
if(!is.null(phyloseq::tax_table(ps_rel, errorIfNULL = FALSE))){
  tax <- as.data.frame(phyloseq::tax_table(ps_rel))
  tax <- tax[rownames(mat_top), , drop = FALSE]
  row_ann <- data.frame(Genus = tax$Genus %||% "Unassigned",
                        Phylum = tax$Phylum %||% "Unassigned",
                        row.names = rownames(mat_top), check.names = FALSE)
}
pheatmap::pheatmap(
  mat_top,
  scale = "row",
  clustering_distance_rows = "euclidean",
  clustering_distance_cols = "euclidean",
  filename = fpath("heatmap_top50_features.pdf"),
  width = 10, height = 10,
  annotation_row = row_ann
)

# Prevalence vs abundance plot (core-ish view)
prev_abund <- data.frame(
  Prevalence = rowMeans(rel_mat > 0),
  MeanAbundance = rowMeans(rel_mat),
  FeatureID = rownames(rel_mat)
)
p_prev <- ggplot(prev_abund, aes(Prevalence, MeanAbundance)) +
  geom_point(alpha = 0.6) +
  scale_y_log10() +
  labs(title = "Prevalence vs Mean Relative Abundance",
       x = "Prevalence (fraction of samples)",
       y = "Mean relative abundance (log scale)")
ggsave(fpath("prevalence_vs_abundance.png"), p_prev, width = 7, height = 6, dpi = 300)

# ---- Export long tables useful downstream ----
# Long-format relative abundance with taxonomy
message("Exporting long-format tables...")
long_rel <- phyloseq::psmelt(ps_rel)
data.table::fwrite(long_rel, fpath("relative_abundance_long.csv"))

# ---- Done ----
message("All done! Results in: ", normalizePath(outdir))
