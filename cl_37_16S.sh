#!/bin/bash
#SBATCH --job-name=16S_metagenomics
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=16S_pipeline_%j.out
#SBATCH --error=16S_pipeline_%j.err

# Exit immediately if any command fails
set -euo pipefail

# Usage Function
usage(){
  echo "Usage: $0 -1 <R1.fastq.gz> -2 <R2.fastq.gz> -o <output_directory> [-t threads] [-a adapters.fa] [-r ref_taxonomy.fa]"
  echo " -1           Forward paired-end reads (gzipped)"
  echo " -2           Reverse paired-end reads (gzipped)"
  echo " -o           Output directory"
  echo " -t           Number of threads (default: 16)"
  echo " -a           Adapter file for trimming (default: TruSeq3-PE.fa)"
  echo " -r           Reference taxonomy database (e.g. SILVA or Greengenes) for taxonomy assignment"
  exit 1
}

# Default parameters
THREADS=16
ADAPTERS="TruSeq3-PE.fa"  # Provide full path if necessary.
REF_TAXONOMY="ref_taxonomy.fa"  # Your reference database in fasta format for taxonomy assignment.

# Parse options
while getopts "1:2:o:t:a:r:" opt; do
  case ${opt} in
    1) READ1=${OPTARG};;
    2) READ2=${OPTARG};;
    o) OUTDIR=${OPTARG};;
    t) THREADS=${OPTARG};;
    a) ADAPTERS=${OPTARG};;
    r) REF_TAXONOMY=${OPTARG};;
    *) usage;;
  esac
done
shift $((OPTIND -1))

# Check required arguments
if [[ -z "${READ1:-}" || -z "${READ2:-}" || -z "${OUTDIR:-}" ]]; then
  usage
fi

# Create output directories for each step
mkdir -p "${OUTDIR}"/{trimmed,merged,otutable,taxonomy,functional,qc,logs}

echo "Using ${THREADS} threads for all multi-threaded tools."

#######################################
## Step 1: Quality Trimming with Trimmomatic
#######################################
echo "Starting quality trimming with Trimmomatic..."
# Here we run Trimmomatic in paired-end mode.
# Accuracy is prioritized by using a sliding window cutoff of 4 bases with an average quality of 20 and a minimum length of 100.
trimmomatic PE -threads "${THREADS}" \
  "${READ1}" "${READ2}" \
  "${OUTDIR}/trimmed/forward.trimmed.fastq.gz" "${OUTDIR}/trimmed/forward.unpaired.fastq.gz" \
  "${OUTDIR}/trimmed/reverse.trimmed.fastq.gz" "${OUTDIR}/trimmed/reverse.unpaired.fastq.gz" \
  ILLUMINACLIP:"${ADAPTERS}":2:30:10 SLIDINGWINDOW:4:20 MINLEN:100
echo "Trimming complete." [7]

#######################################
## Step 2: Merge Paired Reads with PEAR
#######################################
echo "Merging trimmed reads with PEAR..."
# PEAR merges overlapping paired-end reads and is optimized for speed and accuracy.
pear -f "${OUTDIR}/trimmed/forward.trimmed.fastq.gz" \
     -r "${OUTDIR}/trimmed/reverse.trimmed.fastq.gz" \
     -o "${OUTDIR}/merged/merged" \
     -j "${THREADS}" \
     2>&1 | tee "${OUTDIR}/logs/pear.log"
echo "Merging complete." [3][8]

#######################################
## Step 3: Chimera Removal and OTU Clustering using VSEARCH
#######################################
echo "Converting merged fastq to fasta..."
# Convert merged FASTQ to FASTA (if not already in fasta format)
zcat "${OUTDIR}/merged/merged.assembled.fastq.gz" | \
  awk 'NR%4==1 {sub(/^@/,">"); print} NR%4==2 {print}' > "${OUTDIR}/merged/merged.fasta"

echo "Removing chimeras with VSEARCH..."
vsearch --uchime_denovo "${OUTDIR}/merged/merged.fasta" \
  --nonchimeras "${OUTDIR}/merged/merged.nonchimera.fasta" \
  --threads "${THREADS}" \
  --log "${OUTDIR}/logs/vsearch_uchime.log"
echo "Chimera removal complete."

echo "Clustering OTUs at 97% similarity..."
vsearch --cluster_fast "${OUTDIR}/merged/merged.nonchimera.fasta" \
  --id 0.97 \
  --centroids "${OUTDIR}/otutable/otus.fasta" \
  --threads "${THREADS}" \
  --relabel OTU_ \
  --uc "${OUTDIR}/logs/otu_clusters.uc"
echo "OTU clustering done." [9]

#######################################
## Step 4: Taxonomic Assignment
#######################################
echo "Assigning taxonomy..."
# This step uses a taxonomy assignment tool. For example, the QIIME assign_taxonomy.py (or RDP classifier) can be used.
# Make sure the reference taxonomy database is provided (in REF_TAXONOMY)
assign_taxonomy.py -i "${OUTDIR}/otutable/otus.fasta" \
                   -r "${REF_TAXONOMY}" \
                   -o "${OUTDIR}/taxonomy" \
                   --num_threads "${THREADS}" \
                   --confidence 0.8 2>&1 | tee "${OUTDIR}/logs/taxonomy_assignment.log"
echo "Taxonomy assignment complete."

#######################################
## Step 5: Functional Profiling with PICRUSt (or similar)
#######################################
echo "Predicting functional profile with PICRUSt..."
# Generate an OTU table prior to functional prediction. This example assumes that a separate script or command creates an OTU table.
# For instance, you could use biom-format to convert your OTU clusters to a table.
# Here we assume an OTU table called otu_table.biom is created.
# Then run PICRUSt to predict KEGG Ortholog abundances.
predict_metagenomes.py -i "${OUTDIR}/otutable/otu_table.biom" \
                       -o "${OUTDIR}/functional/predicted_metagenomes.txt" \
                       --processes "${THREADS}" 2>&1 | tee "${OUTDIR}/logs/picrust.log"
echo "Functional prediction complete."

#######################################
## Optional: Quality Control Reports (FastQC)
#######################################
echo "Running FastQC on trimmed reads..."
fastqc -t "${THREADS}" "${OUTDIR}/trimmed/"*".fastq.gz" -o "${OUTDIR}/qc"
echo "FastQC complete."

echo "16S Metagenomics pipeline finished successfully."
