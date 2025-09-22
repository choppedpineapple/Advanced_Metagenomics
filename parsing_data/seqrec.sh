#!/usr/bin/env bash
set -euo pipefail
# reconstruct_800bp.sh
# Usage: ./reconstruct_800bp.sh sample_R1.fastq.gz sample_R2.fastq.gz sample_name

R1="$1"
R2="$2"
SAMPLE="${3:-sample}"
THREADS=8

# === Edit these paths / params ===
PRIMER_FASTA="primers.fasta"    # primer sequences (if available)
V_REF="v_reference.fasta"      # V gene ref (optional; helps AssemblePairs)
MIN_CONS_COUNT=2               # min reads per UMI to make consensus
CONS_FREQ=0.6                  # base majority freq for consensus
CLUSTER_ID=0.99                # clustering identity for vsearch
MIN_CLUSTER_SIZE=3             # drop clusters smaller than this
# =================================

workdir="${SAMPLE}_reconstruct"
mkdir -p "$workdir"
cd "$workdir"

echo "### 1) QC and trim (fastp)"
fastp -i "$R1" -I "$R2" -o "${SAMPLE}_R1.trim.fastq.gz" -O "${SAMPLE}_R2.trim.fastq.gz" \
      --detect_adapter_for_pe --trim_poly_g --thread ${THREADS} --quiet

echo "### 2) pRESTO: mask primers (if provided)"
if [[ -s "../$PRIMER_FASTA" ]]; then
  MaskPrimers.py align -s "${SAMPLE}_R1.trim.fastq.gz" "${SAMPLE}_R2.trim.fastq.gz" \
      -p "../$PRIMER_FASTA" --outname "${SAMPLE}.mask" --coord illumina --nproc ${THREADS}
  R1_mask="${SAMPLE}.mask_1_primers-pass.fastq"
  R2_mask="${SAMPLE}.mask_2_primers-pass.fastq"
else
  echo "No primers.fasta found at ../$PRIMER_FASTA — skipping MaskPrimers"
  cp "${SAMPLE}_R1.trim.fastq.gz" "${SAMPLE}.mask_1_primers-pass.fastq"
  cp "${SAMPLE}_R2.trim.fastq.gz" "${SAMPLE}.mask_2_primers-pass.fastq"
  R1_mask="${SAMPLE}.mask_1_primers-pass.fastq"
  R2_mask="${SAMPLE}.mask_2_primers-pass.fastq"
fi

echo "### 3) pRESTO: pair mates (copy metadata like UMIs between mates)"
PairSeq.py -1 "$R1_mask" -2 "$R2_mask" --outname "${SAMPLE}.paired" --nproc ${THREADS}
paired1="${SAMPLE}.paired_1.fastq"
paired2="${SAMPLE}.paired_2.fastq"

echo "### 4) pRESTO: assemble pairs (sequential: de-novo then reference-guided if V_REF available)"
if [[ -s "../$V_REF" ]]; then
  AssemblePairs.py sequential -1 "$paired1" -2 "$paired2" -r "../$V_REF" --coord illumina \
      --outname "${SAMPLE}.assembled" --nproc ${THREADS}
else
  AssemblePairs.py align -1 "$paired1" -2 "$paired2" --coord illumina \
      --outname "${SAMPLE}.assembled" --nproc ${THREADS}
fi
merged_fastq="${SAMPLE}.assembled_assemble-pass.fastq"
failed_fastq="${SAMPLE}.assembled_assemble-fail.fastq"

echo "### 5) Optional: merge leftover unassembled mates with BBMerge (improve merged yield)"
if command -v bbmerge.sh &>/dev/null; then
  bbmerge.sh in1="$paired1" in2="$paired2" out="${SAMPLE}.bbmerged.fastq" outu1="${SAMPLE}.bb_unmerged_1.fastq" \
            outu2="${SAMPLE}.bb_unmerged_2.fastq" strict=t minoverlap=20 threads=${THREADS}
  # combine with pRESTO merged
  cat "${SAMPLE}.bbmerged.fastq" >> "$merged_fastq"
fi

echo "### 6) Quality filter merged reads (pRESTO FilterSeq)"
FilterSeq.py -s "$merged_fastq" --minlen 50 --minqual 20 --outname "${SAMPLE}.merged.filtered"

filtered_fastq="${SAMPLE}.merged.filtered_pass.fastq"

echo "### 7) pRESTO: Build consensus by UMI (if UMIs exist). If no UMIs, this will cluster by chosen barcode field (BF)."
# Replace BARCODE with the actual annotation name used by your pipeline for UMI (e.g., 'UMI' or 'BC')
# If no UMI, you can skip BuildConsensus and use dereplication on merged reads instead.
BuildConsensus.py -s "$filtered_fastq" --bf BARCODE --min_count ${MIN_CONS_COUNT} --freq ${CONS_FREQ} \
                  --outname "${SAMPLE}.consensus" --nproc ${THREADS} || \
  (echo "BuildConsensus failed or no BARCODE field — skipping consensus step." && cp "$filtered_fastq" "${SAMPLE}.consensus_consensus-pass.fastq")

consensus_fastq="${SAMPLE}.consensus_consensus-pass.fastq"

echo "### 8) Convert consensus FASTQ -> FASTA and dereplicate (vsearch)"
# Convert to fasta
SeqIOConvert.py -s "$consensus_fastq" --fasta --outname "${SAMPLE}.consensus_fa"
consensus_fa="${SAMPLE}.consensus_fa.fasta"

# Dereplicate exact sequences (keeps abundance in headers)
vsearch --derep_fulllength "$consensus_fa" --output "${SAMPLE}.uniques.fasta" --sizeout --minuniquesize 1 --threads ${THREADS}

echo "### 9) Remove denovo chimeras"
vsearch --uchime_denovo "${SAMPLE}.uniques.fasta" --nonchimeras "${SAMPLE}.nonchim.fasta" --threads ${THREADS}

echo "### 10) Cluster at high identity to separate near-identical clones"
vsearch --cluster_fast "${SAMPLE}.nonchim.fasta" --id ${CLUSTER_ID} --centroids "${SAMPLE}.centroids.fasta" \
        --uc "${SAMPLE}.clusters.uc" --threads ${THREADS}

echo "### 11) Filter small clusters (optional, drop low-abundance clusters)"
# keep clusters with size >= MIN_CLUSTER_SIZE (size encoded as ;size= in header)
awk -v min=${MIN_CLUSTER_SIZE} 'BEGIN{RS=">"; ORS=""} NR>1{if(match($0,/size=([0-9]+)/,a) && a[1]+0>=min) print ">"$0}' "${SAMPLE}.centroids.fasta" > "${SAMPLE}.centroids.filtered.fasta"

echo "### 12) For each cluster: gather reads mapping to centroid (from earlier consensus FASTA) and assemble with CAP3"
# Map consensus reads to centroids to gather cluster membership
vsearch --usearch_global "$consensus_fa" --db "${SAMPLE}.centroids.filtered.fasta" --id ${CLUSTER_ID} \
        --uc "${SAMPLE}.reads_vs_centroids.uc" --strand both --threads ${THREADS}

# create per-cluster fasta and CAP3 assemble (simple loop)
mkdir -p clusters caps
awk '/^>/{name=$0;getline;seq[$0]=name} END{print ""}' "$consensus_fa" >/dev/null 2>&1 || true

# parse UC file to gather read IDs per centroid
awk -v CONS="$consensus_fa" -F'\t' '$1=="S" || $1=="H" {cent=$10; read=$9; print cent"\t"read}' "${SAMPLE}.reads_vs_centroids.uc" \
  | sort | awk '{print > ("clusters/"$1".txt") }'

for f in clusters/*.txt; do
  cluster=$(basename "$f" .txt)
  seqtk subseq "$consensus_fa" "$f" > "clusters/${cluster}.fasta"
  # require at least 2 sequences to assemble
  cnt=$(grep -c "^>" "clusters/${cluster}.fasta" || true)
  if [[ "$cnt" -ge 2 ]]; then
    cap3 "clusters/${cluster}.fasta" > "caps/${cluster}.cap.log" || true
    # collect contigs
    if [[ -f "clusters/${cluster}.fasta.cap.contigs" ]]; then
      cat "clusters/${cluster}.fasta.cap.contigs" >> "${SAMPLE}.all_contigs.fasta"
    else
      # fallback: use centroid sequence
      grep -A1 "^>" "clusters/${cluster}.fasta" | sed '/^--/d' >> "${SAMPLE}.all_contigs.fasta"
    fi
  else
    # singletons: keep as-is
    cat "clusters/${cluster}.fasta" >> "${SAMPLE}.all_contigs.fasta"
  fi
done

echo "### 13) Optional: iterative seed-and-extend for fragmented contigs using PRICE"
# Example for one seed contig; you can loop on caps/*.cap.contigs
# price -ic seed.fasta 250 1 -icq reads.fastq 250 1 -o price_out -tp ${THREADS}
# (left as optional—PRICE params depend heavily on your data)

echo "### 14) Validate contigs with IgBlast (requires IgBlast db setup)"
if command -v igblastn &>/dev/null; then
  igblastn -germline_db_V ../igblast_db/V -germline_db_D ../igblast_db/D -germline_db_J ../igblast_db/J \
           -query "${SAMPLE}.all_contigs.fasta" -outfmt 7 -out "${SAMPLE}.igblast.out" || true
  echo "IgBlast results: ${SAMPLE}.igblast.out"
else
  echo "igblastn not found: skipping IgBlast validation"
fi

echo "### DONE. Outputs:"
echo " - cleaned merged/consensus reads: ${consensus_fastq}"
echo " - assembled contigs: ${SAMPLE}.all_contigs.fasta"
echo " - igblast output (if run): ${SAMPLE}.igblast.out"
