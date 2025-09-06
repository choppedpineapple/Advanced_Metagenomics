#!/usr/bin/env Rscript

# =========================
# Non-phylogenetic QC + Diversity + Abundance
# Inputs : table.qza (feature table), taxonomy.qza
# Outputs: QC summaries, alpha/beta diversity, top taxa plots
# Usage  : Rscript qiime2_alpha_beta_qc.R table.qza taxonomy.qza output_dir
# =========================

# ---- Args ----
args <- commandArgs(trailingOnly = TRUE)
table_fp   <- ifelse(length(args) >= 1, args[1], "table.qza")
tax_fp     <- ifelse(length(args) >= 2, args[2], "taxonomy.qza")
outdir     <- ifelse(length(args) >= 3, args[3], "qiime2_r_results")

# ---- Packages ----
req_pkgs <- c("qiime2R","phyloseq","tidyverse","vegan","data.table","pheatmap","patchwork","RColorBrewer")
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

# ---- Import ----
message("Reading QIIME2 artifacts...")
ps <- qiime2R::qza_to_phyloseq(features = table_fp, taxonomy = tax_fp)

# Ensure minimal sample_data
if(is.null(sample_data(ps, errorIfNULL = FALSE))){
  samp_ids <- sample_names(ps)
  libsize  <- colSums(otu_table(ps))
  meta_df  <- data.frame(Sample = samp_ids, LibrarySize = libsize, row.names = samp_ids, check.names = FALSE)
  sample_data(ps) <- sample_data(meta_df)
}

# ---- Taxonomy cleanup ----
if(!is.null(tax_table(ps, errorIfNULL = FALSE))){
  tax <- as.data.frame(tax_table(ps))
  tax[is.na(tax) | tax == ""] <- "Unassigned"
  tax_table(ps) <- as.matrix(tax)
}

# ---- Filtering ----
message("Filtering unwanted taxa...")
if(!is.null(tax_table(ps, errorIfNULL = FALSE))){
  tax <- as.data.frame(tax_table(ps))
  bad_patterns <- c("Chloroplast", "Mitochondria", "Eukaryota", "Unassigned", "NA")
  keep_idx <- apply(tax, 1, function(row) {
    all(!is.na(row)) &&
      !any(sapply(bad_patterns, function(p) any(grepl(p, row, ignore.case = TRUE))))
  })
  ps <- prune_taxa(keep_idx, ps)
}

# abundance & prevalence filters
min_prev <- 0.05 * nsamples(ps)     # present in ≥5% of samples
min_abund <- 0.001                  # mean relative abundance ≥0.1%
rel_abund <- taxa_sums(ps) / sum(taxa_sums(ps))
prev <- apply(otu_table(ps) > 0, 1, sum)
keep_idx2 <- (prev >= min_prev) | (rel_abund >= min_abund)
ps <- prune_taxa(keep_idx2, ps)

message("After filtering: ", ntaxa(ps), " taxa remain.")

# ---- Relative abundance ----
ps_rel <- transform_sample_counts(ps, function(x) x / sum(x))

# ---- Alpha diversity ----
alpha_df <- estimate_richness(ps, measures = c("Observed","Shannon","Simpson","InvSimpson"))
alpha_df$Evenness_Pielou <- with(alpha_df, ifelse(Observed > 0, Shannon / log(Observed), NA))
alpha_df$Sample <- rownames(alpha_df)
alpha_df <- cbind(alpha_df, data.frame(sample_data(ps)[alpha_df$Sample, , drop = FALSE]))
fwrite(alpha_df, fpath("alpha_diversity.csv"))

p1 <- ggplot(alpha_df, aes(LibrarySize, Observed)) +
  geom_point() + geom_smooth(method="loess") +
  labs(title="Observed Features vs Library Size", x="Library size", y="Observed")
p2 <- ggplot(alpha_df, aes(LibrarySize, Shannon)) +
  geom_point() + geom_smooth(method="loess") +
  labs(title="Shannon Diversity vs Library Size")
p3 <- ggplot(alpha_df, aes(LibrarySize, Evenness_Pielou)) +
  geom_point() + geom_smooth(method="loess") +
  labs(title="Pielou's Evenness vs Library Size")

ggsave(fpath("alpha_vs_depth.pdf"), (p1/p2/p3), width=8, height=12, device="pdf")

# Rarefaction
pdf(fpath("rarefaction_curves.pdf"), width=8, height=6)
vegan::rarecurve(t(as(otu_table(ps), "matrix")), step=100, cex=0.6, label=TRUE)
dev.off()

# ---- Beta diversity ----
dist_bray <- phyloseq::distance(ps, method="bray")
dist_jacc <- phyloseq::distance(ps, method="jaccard", binary=TRUE)
fwrite(as.data.frame(as.matrix(dist_bray)), fpath("beta_bray_curtis.tsv"), sep="\t")
fwrite(as.data.frame(as.matrix(dist_jacc)), fpath("beta_jaccard.tsv"), sep="\t")

ord_bray <- ordinate(ps, method="PCoA", distance=dist_bray)
ord_jacc <- ordinate(ps, method="PCoA", distance=dist_jacc)

ggsave(fpath("pcoa_bray.pdf"), plot_ordination(ps, ord_bray) + geom_point(size=3), width=7, height=6, device="pdf")
ggsave(fpath("pcoa_jaccard.pdf"), plot_ordination(ps, ord_jacc) + geom_point(size=3), width=7, height=6, device="pdf")

# ---- Top-N plots ----
plot_topN_rank <- function(ps_obj, rank, topN, file_prefix){
  if(is.null(tax_table(ps_obj, errorIfNULL=FALSE))) return(NULL)
  ps_glom <- tryCatch(tax_glom(ps_obj, taxrank=rank, NArm=FALSE), error=function(e) NULL)
  if(is.null(ps_glom)) return(NULL)

  df <- psmelt(ps_glom) %>% mutate(Taxon = get(rank))
  df$Taxon[is.na(df$Taxon) | df$Taxon==""] <- "Unassigned"

  top_taxa <- df %>% group_by(Taxon) %>%
    summarise(MeanAbund=mean(Abundance, na.rm=TRUE)) %>%
    arrange(desc(MeanAbund)) %>% slice_head(n=topN) %>% pull(Taxon)

  df$TaxonCollapsed <- ifelse(df$Taxon %in% top_taxa, as.character(df$Taxon), "Other")

  p <- ggplot(df, aes(x=Sample, y=Abundance, fill=TaxonCollapsed)) +
    geom_bar(stat="identity") +
    labs(title=paste("Relative Abundance @", rank), y="Relative abundance") +
    theme(axis.text.x=element_text(angle=90, vjust=0.5))

  ggsave(fpath(paste0(file_prefix,"_stacked_bar.pdf")), p, width=12, height=6, device="pdf")

  mean_tbl <- df %>% group_by(Taxon) %>%
    summarise(MeanRelAbundance=mean(Abundance, na.rm=TRUE)) %>%
    arrange(desc(MeanRelAbundance))
  fwrite(mean_tbl, fpath(paste0(file_prefix,"_mean_rel_abundance.csv")))
}

plot_topN_rank(ps_rel, rank="Phylum", topN=15, file_prefix="phylum")
plot_topN_rank(ps_rel, rank="Genus",  topN=20, file_prefix="genus")

# ---- Heatmap ----
rel_mat <- as.matrix(otu_table(ps_rel))
topN <- min(50, nrow(rel_mat))
top_idx <- order(rowMeans(rel_mat), decreasing=TRUE)[1:topN]
mat_top <- rel_mat[top_idx,,drop=FALSE]

row_ann <- NULL
if(!is.null(tax_table(ps_rel, errorIfNULL=FALSE))){
  tax <- as.data.frame(tax_table(ps_rel))
  row_ann <- data.frame(Genus = tax$Genus[top_idx], Phylum = tax$Phylum[top_idx],
                        row.names=rownames(mat_top))
}

pheatmap(mat_top, scale="row", clustering_distance_rows="euclidean",
         clustering_distance_cols="euclidean", annotation_row=row_ann,
         filename=fpath("heatmap_top50_features.pdf"), width=10, height=10)

# ---- Done ----
message("Analysis complete. Results in: ", normalizePath(outdir))
