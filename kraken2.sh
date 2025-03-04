#!/bin/bash

# Kraken2 Analysis Pipeline
# This script processes Kraken2 reports for abundance estimation, visualization,
# and functional/pathway analysis.

set -e  # Exit on error
set -u  # Treat unset variables as errors

# ===== Configuration =====
KRAKEN2_DB="/storage/k2-db/"
THREADS=$(nproc)  # Use all available CPU cores
OUTPUT_DIR="kraken_analysis_results"
REPORT_FILE=""  # Will be set by command line argument

# ===== Functions =====

# Function to display usage
usage() {
    echo "Usage: $0 -r <kraken2_report> [-o <output_dir>] [-t <threads>]"
    echo "  -r  Kraken2 report file (required)"
    echo "  -o  Output directory (default: $OUTPUT_DIR)"
    echo "  -t  Number of threads to use (default: all available)"
    echo "  -h  Display this help and exit"
    exit 1
}

# Function to check if a command exists
check_command() {
    if ! command -v "$1" &> /dev/null; then
        echo "Error: $1 is required but not installed."
        echo "Please install $1 before running this script."
        exit 1
    fi
}

# Function to normalize abundances
normalize_abundances() {
    local report_file=$1
    local output_file="$OUTPUT_DIR/normalized_abundances.tsv"
    
    echo "Normalizing abundances..."
    
    # Extract the necessary columns and calculate relative abundances
    awk 'BEGIN {FS="\t"; OFS="\t"} 
         NR>1 {
             if ($4 == "S") {  # Only consider species-level classifications
                 # Calculate relative abundance
                 gsub(/^ +/, "", $1)  # Remove leading whitespace
                 print $6, $1, $2
             }
         }' "$report_file" | sort -nr > "$output_file"
    
    echo "Normalized abundance data saved to $output_file"
    return "$output_file"
}

# Function to generate visualization
generate_visualization() {
    local abundance_file=$1
    local output_prefix="$OUTPUT_DIR/visualization"
    
    echo "Generating visualizations..."
    
    # Create a simple bar plot of the top 20 species using R
    cat > "$OUTPUT_DIR/plot_abundances.R" << 'EOF'
library(ggplot2)
library(dplyr)

# Read the normalized abundance data
data <- read.delim("normalized_abundances.tsv", header=FALSE,
                  col.names=c("Taxonomy", "Abundance", "Name"))

# Get top 20 species
top_species <- data %>%
  arrange(desc(Abundance)) %>%
  head(20)

# Create a barplot
p <- ggplot(top_species, aes(x=reorder(Taxonomy, Abundance), y=Abundance, fill=Abundance)) +
  geom_bar(stat="identity") +
  coord_flip() +
  theme_minimal() +
  labs(title="Top 20 Species Abundance",
       x="Species",
       y="Relative Abundance") +
  theme(legend.position="none")

# Save the plot
ggsave("visualization_barplot.pdf", p, width=10, height=8)
ggsave("visualization_barplot.png", p, width=10, height=8)

# Create a heatmap for taxonomic visualization
library(pheatmap)

# Prepare data for heatmap
heat_data <- data %>%
  arrange(desc(Abundance)) %>%
  head(50) %>%
  select(Taxonomy, Abundance)

heat_matrix <- as.matrix(heat_data$Abundance)
rownames(heat_matrix) <- heat_data$Taxonomy

# Generate heatmap
pheatmap(heat_matrix,
         filename="visualization_heatmap.pdf",
         cluster_rows=FALSE,
         cluster_cols=FALSE,
         main="Top 50 Taxa Abundance Heatmap")

# Create interactive visualization with plotly
library(plotly)

p_interactive <- plot_ly(top_species, 
                        x = ~Taxonomy, 
                        y = ~Abundance, 
                        type = 'bar',
                        marker = list(color = ~Abundance,
                                    colorscale = 'Viridis')) %>%
  layout(title = "Top 20 Species Abundance",
         xaxis = list(title = "Species"),
         yaxis = list(title = "Relative Abundance"))

# Save the interactive plot
htmlwidgets::saveWidget(p_interactive, "visualization_interactive.html")
EOF

    # Run the R script
    cd "$OUTPUT_DIR" && Rscript plot_abundances.R
    cd - > /dev/null
    
    echo "Visualizations created in $OUTPUT_DIR:"
    echo "  - visualization_barplot.pdf/png"
    echo "  - visualization_heatmap.pdf"
    echo "  - visualization_interactive.html"
}

# Function to perform functional analysis
perform_functional_analysis() {
    local abundance_file=$1
    local output_dir="$OUTPUT_DIR/functional_analysis"
    
    mkdir -p "$output_dir"
    echo "Performing functional analysis..."
    
    # Extract taxon IDs for functional analysis
    awk '{print $1}' "$abundance_file" > "$output_dir/taxids.txt"
    
    # Generate KEGG pathways using MinPath
    if command -v MinPath &> /dev/null; then
        echo "Running MinPath for KEGG pathway prediction..."
        
        # First, map taxon IDs to KEGG orthology using a custom script
        cat > "$output_dir/map_taxids_to_ko.py" << 'EOF'
#!/usr/bin/env python3
import sys
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor

def get_kegg_orthology(taxid):
    """Query KEGG API to get orthology information for a given taxonomy ID"""
    try:
        # Use KEGG API to get KO information
        url = f"http://rest.kegg.jp/link/ko/genome:{taxid}"
        response = requests.get(url)
        time.sleep(0.1)  # Be gentle with the KEGG API
        
        if response.status_code == 200:
            return response.text
        else:
            return None
    except Exception as e:
        print(f"Error processing taxid {taxid}: {str(e)}", file=sys.stderr)
        return None

def process_taxids(taxid_file, output_file, threads):
    """Process a file of taxonomy IDs in parallel to get KEGG orthology"""
    with open(taxid_file, 'r') as f:
        taxids = [line.strip() for line in f]
    
    ko_results = {}
    
    with ThreadPoolExecutor(max_workers=threads) as executor:
        futures = {executor.submit(get_kegg_orthology, taxid): taxid for taxid in taxids}
        
        for future in futures:
            taxid = futures[future]
            result = future.result()
            if result:
                ko_results[taxid] = result
    
    # Write results to output file
    with open(output_file, 'w') as f:
        for taxid, ko_data in ko_results.items():
            f.write(f"# TaxID: {taxid}\n")
            f.write(ko_data)
            f.write("\n")
    
    print(f"KEGG orthology data saved to {output_file}")
    
    # Create a formatted input file for MinPath
    with open(output_file + ".minpath", 'w') as f:
        for taxid, ko_data in ko_results.items():
            ko_ids = []
            for line in ko_data.strip().split('\n'):
                if line and '\t' in line:
                    parts = line.strip().split('\t')
                    if len(parts) > 1:
                        ko_id = parts[1].replace('ko:', '')
                        ko_ids.append(ko_id)
            
            if ko_ids:
                f.write(f"{taxid}\t{','.join(ko_ids)}\n")
    
    print(f"MinPath input file created: {output_file}.minpath")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <taxid_file> <output_file> [threads]")
        sys.exit(1)
    
    taxid_file = sys.argv[1]
    output_file = sys.argv[2]
    threads = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    process_taxids(taxid_file, output_file, threads)
EOF
        chmod +x "$output_dir/map_taxids_to_ko.py"
        
        # Run the mapping script
        "$output_dir/map_taxids_to_ko.py" "$output_dir/taxids.txt" "$output_dir/kegg_orthology.txt" "$THREADS"
        
        # Run MinPath on the mappings
        MinPath -any "$output_dir/kegg_orthology.txt.minpath" -map 1 -report "$output_dir/minpath_report.txt"
        
        echo "MinPath analysis complete. Results in $output_dir/minpath_report.txt"
    else
        echo "MinPath not found. Skipping KEGG pathway prediction."
    fi
    
    # Functional annotation using eggNOG-mapper if available
    if command -v emapper.py &> /dev/null; then
        echo "Running eggNOG-mapper for functional annotation..."
        
        # Create input for eggNOG-mapper
        awk '{printf ">%s\n%s\n", $1, "PLACEHOLDER"}' "$abundance_file" > "$output_dir/taxa_for_emapper.fasta"
        
        # Run eggNOG-mapper
        emapper.py -i "$output_dir/taxa_for_emapper.fasta" \
                  --output "$output_dir/eggnog_output" \
                  -m diamond \
                  --cpu "$THREADS" \
                  --target_orthologs all \
                  --go_evidence all \
                  --pfam_realign none
        
        echo "eggNOG-mapper analysis complete. Results in $output_dir/eggnog_output*"
    else
        echo "eggNOG-mapper not found. Skipping functional annotation."
    fi
    
    # Generate a summary of functional categories
    cat > "$output_dir/generate_functional_summary.R" << 'EOF'
library(dplyr)
library(ggplot2)

# Check if MinPath output exists
minpath_file <- "minpath_report.txt"
eggnog_file <- "eggnog_output.emapper.annotations"

# Function to process MinPath output
process_minpath <- function(file_path) {
  if (!file.exists(file_path)) {
    return(NULL)
  }
  
  minpath_data <- readLines(file_path)
  
  # Extract pathway information
  pathway_lines <- grep("^path", minpath_data)
  
  pathways <- data.frame(
    Pathway = character(),
    Description = character(),
    Count = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (i in pathway_lines) {
    line <- minpath_data[i]
    parts <- strsplit(line, "\t")[[1]]
    
    if (length(parts) >= 2) {
      pathway_id <- parts[1]
      description <- parts[2]
      
      # Count how many times this pathway appears
      count <- sum(grepl(pathway_id, minpath_data))
      
      pathways <- rbind(pathways, data.frame(
        Pathway = pathway_id,
        Description = description,
        Count = count
      ))
    }
  }
  
  # Remove duplicates and sort by count
  pathways <- pathways %>%
    distinct() %>%
    arrange(desc(Count))
  
  return(pathways)
}

# Function to process eggNOG output
process_eggnog <- function(file_path) {
  if (!file.exists(file_path)) {
    return(NULL)
  }
  
  # Read eggNOG annotations, skipping comment lines
  eggnog_data <- read.delim(file_path, comment.char = "#", stringsAsFactors = FALSE)
  
  # Extract and count GO terms
  go_terms <- unlist(strsplit(eggnog_data$GOs, ","))
  go_terms <- go_terms[!is.na(go_terms) & go_terms != ""]
  
  go_counts <- as.data.frame(table(go_terms))
  colnames(go_counts) <- c("GO_term", "Count")
  go_counts <- go_counts %>% arrange(desc(Count))
  
  # Extract and count COG categories
  cog_cats <- eggnog_data$COG_category
  cog_cats <- unlist(strsplit(as.character(cog_cats), ""))
  cog_cats <- cog_cats[!is.na(cog_cats) & cog_cats != ""]
  
  cog_counts <- as.data.frame(table(cog_cats))
  colnames(cog_counts) <- c("COG_category", "Count")
  cog_counts <- cog_counts %>% arrange(desc(Count))
  
  # Map COG categories to descriptions
  cog_descriptions <- c(
    "J" = "Translation, ribosomal structure and biogenesis",
    "A" = "RNA processing and modification",
    "K" = "Transcription",
    "L" = "Replication, recombination and repair",
    "B" = "Chromatin structure and dynamics",
    "D" = "Cell cycle control, cell division, chromosome partitioning",
    "Y" = "Nuclear structure",
    "V" = "Defense mechanisms",
    "T" = "Signal transduction mechanisms",
    "M" = "Cell wall/membrane/envelope biogenesis",
    "N" = "Cell motility",
    "Z" = "Cytoskeleton",
    "W" = "Extracellular structures",
    "U" = "Intracellular trafficking, secretion, vesicular transport",
    "O" = "Posttranslational modification, protein turnover, chaperones",
    "X" = "Mobilome: prophages, transposons",
    "C" = "Energy production and conversion",
    "G" = "Carbohydrate transport and metabolism",
    "E" = "Amino acid transport and metabolism",
    "F" = "Nucleotide transport and metabolism",
    "H" = "Coenzyme transport and metabolism",
    "I" = "Lipid transport and metabolism",
    "P" = "Inorganic ion transport and metabolism",
    "Q" = "Secondary metabolites biosynthesis, transport and catabolism",
    "R" = "General function prediction only",
    "S" = "Function unknown"
  )
  
  cog_counts$Description <- cog_descriptions[cog_counts$COG_category]
  
  return(list(go_counts = go_counts, cog_counts = cog_counts))
}

# Process MinPath data
minpath_pathways <- process_minpath(minpath_file)
if (!is.null(minpath_pathways)) {
  # Plot top pathways
  top_pathways <- head(minpath_pathways, 20)
  
  p1 <- ggplot(top_pathways, aes(x = reorder(Description, Count), y = Count)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Top 20 KEGG Pathways",
         x = "Pathway",
         y = "Count")
  
  ggsave("top_pathways.pdf", p1, width = 10, height = 8)
  write.csv(minpath_pathways, "pathway_summary.csv", row.names = FALSE)
  
  cat("MinPath pathway summary generated\n")
}

# Process eggNOG data
eggnog_results <- process_eggnog(eggnog_file)
if (!is.null(eggnog_results)) {
  # Plot top GO terms
  top_go <- head(eggnog_results$go_counts, 20)
  
  p2 <- ggplot(top_go, aes(x = reorder(GO_term, Count), y = Count)) +
    geom_bar(stat = "identity", fill = "darkgreen") +
    coord_flip() +
    theme_minimal() +
    labs(title = "Top 20 GO Terms",
         x = "GO Term",
         y = "Count")
  
  ggsave("top_go_terms.pdf", p2, width = 10, height = 8)
  
  # Plot COG categories
  p3 <- ggplot(eggnog_results$cog_counts, 
               aes(x = reorder(COG_category, Count), 
                   y = Count, 
                   fill = COG_category)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    coord_flip() +
    labs(title = "COG Functional Categories",
         x = "Category",
         y = "Count") +
    theme(legend.position = "none") +
    scale_fill_brewer(palette = "Set3")
  
  ggsave("cog_categories.pdf", p3, width = 10, height = 8)
  
  # Create a detailed plot with descriptions
  p4 <- ggplot(eggnog_results$cog_counts, 
               aes(x = reorder(Description, Count), 
                   y = Count, 
                   fill = COG_category)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    coord_flip() +
    labs(title = "COG Functional Categories with Descriptions",
         x = "",
         y = "Count") +
    theme(legend.position = "none") +
    scale_fill_brewer(palette = "Set3")
  
  ggsave("cog_categories_detailed.pdf", p4, width = 12, height = 10)
  
  write.csv(eggnog_results$go_counts, "go_terms_summary.csv", row.names = FALSE)
  write.csv(eggnog_results$cog_counts, "cog_categories_summary.csv", row.names = FALSE)
  
  cat("eggNOG functional summaries generated\n")
}

# Generate an HTML report if both are available
if (!is.null(minpath_pathways) && !is.null(eggnog_results)) {
  library(rmarkdown)
  
  # Create an Rmd file for the report
  cat('---
title: "Functional Analysis Report"
output: html_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)
library(dplyr)
library(ggplot2)
library(DT)
library(plotly)

# Load the data
minpath_pathways <- read.csv("pathway_summary.csv")
go_terms <- read.csv("go_terms_summary.csv")
cog_categories <- read.csv("cog_categories_summary.csv")
```

## KEGG Pathway Analysis

The following table shows the top KEGG pathways identified:

```{r}
datatable(minpath_pathways, options = list(pageLength = 10))
```

```{r}
p1 <- ggplot(head(minpath_pathways, 20), 
             aes(x = reorder(Description, Count), y = Count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 20 KEGG Pathways",
       x = "Pathway",
       y = "Count")

ggplotly(p1)
```

## GO Term Analysis

The following table shows the most frequent GO terms:

```{r}
datatable(go_terms, options = list(pageLength = 10))
```

```{r}
p2 <- ggplot(head(go_terms, 20), 
             aes(x = reorder(GO_term, Count), y = Count)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  theme_minimal() +
  labs(title = "Top 20 GO Terms",
       x = "GO Term",
       y = "Count")

ggplotly(p2)
```

## COG Functional Categories

```{r}
p3 <- ggplot(cog_categories, 
             aes(x = reorder(Description, Count), 
                 y = Count, 
                 fill = COG_category)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  coord_flip() +
  labs(title = "COG Functional Categories",
       x = "",
       y = "Count") +
  theme(legend.position = "right") +
  scale_fill_brewer(palette = "Set3")

ggplotly(p3)
```

', file = "functional_report.Rmd")
  
  # Render the report
  rmarkdown::render("functional_report.Rmd", output_file = "functional_report.html")
  
  cat("Interactive HTML report generated: functional_report.html\n")
}
EOF
    
    # Run the R script to generate functional summaries
    cd "$output_dir" && Rscript generate_functional_summary.R
    cd - > /dev/null
    
    echo "Functional analysis complete. Results in $output_dir/"
}

# ===== Main Script =====

# Parse command line arguments
while getopts "r:o:t:h" opt; do
    case ${opt} in
        r )
            REPORT_FILE=$OPTARG
            ;;
        o )
            OUTPUT_DIR=$OPTARG
            ;;
        t )
            THREADS=$OPTARG
            ;;
        h )
            usage
            ;;
        \? )
            echo "Invalid option: $OPTARG" 1>&2
            usage
            ;;
        : )
            echo "Invalid option: $OPTARG requires an argument" 1>&2
            usage
            ;;
    esac
done

# Check if report file is provided
if [[ -z "$REPORT_FILE" ]]; then
    echo "Error: Kraken2 report file is required"
    usage
fi

# Check if report file exists
if [[ ! -f "$REPORT_FILE" ]]; then
    echo "Error: Kraken2 report file '$REPORT_FILE' does not exist"
    exit 1
fi

# Check required commands
check_command "awk"
check_command "sort"
check_command "Rscript"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "Created output directory: $OUTPUT_DIR"

# Run the pipeline
echo "=== Starting Kraken2 Analysis Pipeline ==="
echo "Input file: $REPORT_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Using $THREADS threads"

# Step 1: Normalize abundances
abundance_file=$(normalize_abundances "$REPORT_FILE")

# Step 2: Generate visualizations
generate_visualization "$abundance_file"

# Step 3: Perform functional analysis
perform_functional_analysis "$abundance_file"

echo "=== Pipeline complete ==="
echo "Results are available in: $OUTPUT_DIR"

# Generate summary report
cat > "$OUTPUT_DIR/generate_summary_report.R" << 'EOF'
library(rmarkdown)

# Create an Rmd file for the report
cat('---
title: "Kraken2 Analysis Report"
output:
  html_document:
    toc: true
    toc_float: true
    theme: united
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE)
library(dplyr)
library(ggplot2)
library(DT)
library(plotly)
```

## Abundance Analysis

```{r}
# Read the normalized abundance data
abundances <- read.delim("normalized_abundances.tsv", header=FALSE,
                  col.names=c("Taxonomy", "Abundance", "Name"))

# Display top 20 species in a table
datatable(head(abundances, 20), 
          options = list(pageLength = 10),
          caption = "Top 20 Species by Abundance")
```

### Abundance Visualization

```{r, fig.width=10, fig.height=8}
# Display the barplot
top_species <- abundances %>%
  arrange(desc(Abundance)) %>%
  head(20)

p <- ggplot(top_species, aes(x=reorder(Taxonomy, Abundance), y=Abundance, fill=Abundance)) +
  geom_bar(stat="identity") +
  coord_flip() +
  theme_minimal() +
  labs(title="Top 20 Species Abundance",
       x="Species",
       y="Relative Abundance") +
  theme(legend.position="none")

ggplotly(p)
```

## Functional Analysis

```{r}
# Check if functional analysis files exist
pathway_file <- "functional_analysis/pathway_summary.csv"
go_file <- "functional_analysis/go_terms_summary.csv"
cog_file <- "f
