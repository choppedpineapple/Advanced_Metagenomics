#!/bin/bash
set -eoux pipefail # Exit on error, print commands, treat unset variables as error, pipefail

# --- CONFIGURATION (NON-UMI) ---
# YOU MUST ADJUST THESE VARIABLES
SAMPLE_NAME="my_sample_nonumi"
R1_IN="input_R1.fastq.gz" # Your input R1 FASTQ.gz file
R2_IN="input_R2.fastq.gz" # Your input R2 FASTQ.gz file

# Primer FASTA files (if needed for specific trimming beyond adapter removal)
# If your primers are effectively part of the adapters and removed by fastp,
# you might be able to simplify or skip MaskPrimers.
# However, for internal antibody primers, MaskPrimers is often still best.
FWD_PRIMER_FASTA="fwd_primers.fasta" # V-gene primers (expected on R1)
REV_PRIMER_FASTA="rev_primers.fasta" # C-gene/J-gene primers (expected on R2)

# Tool paths (if not in PATH)
# Assuming they are in PATH.

# Resources
THREADS=4

# Output directory
OUT_DIR="${SAMPLE_NAME}_processed_nonumi"
mkdir -p "$OUT_DIR"

# --- SCRIPT START ---
echo "Starting NON-UMI preprocessing for sample: $SAMPLE_NAME"
cd "$OUT_DIR" # Work within the output directory

# 1. Initial QC, Adapter/Quality Trimming, and Paired-End Merging with fastp
# fastp can also perform read merging. If this works well, it's very direct.
echo "Step 1: Running fastp for QC, adapter trimming, and optional merging..."
R1_QC="${SAMPLE_NAME}_R1.qc.fastq.gz"
R2_QC="${SAMPLE_NAME}_R2.qc.fastq.gz"
MERGED_FASTP_OUT="${SAMPLE_NAME}.fastp_merged.fastq.gz" # fastp's merged output
UNPAIRED1_FASTP_OUT="${SAMPLE_NAME}.fastp_unpaired1.fastq.gz"
UNPAIRED2_FASTP_OUT="${SAMPLE_NAME}.fastp_unpaired2.fastq.gz"

fastp \
    -i "../$R1_IN" -I "../$R2_IN" \
    -o "$R1_QC" -O "$R2_QC" \
    --html "${SAMPLE_NAME}.fastp.html" --json "${SAMPLE_NAME}.fastp.json" \
    --length_required 50 \
    --cut_front --cut_front_window_size 1 --cut_front_mean_quality 20 \
    --cut_tail --cut_tail_window_size 1 --cut_tail_mean_quality 20 \
    --trim_poly_g \
    --n_base_limit 5 \
    --average_qual 20 \
    --thread "$THREADS" \
    --merge \
    --merged_out "$MERGED_FASTP_OUT" \
    --out1 "$UNPAIRED1_FASTP_OUT" \
    --out2 "$UNPAIRED2_FASTP_OUT"
    # Check fastp report to see how many reads merged.
    # If merging is highly effective, you might primarily use $MERGED_FASTP_OUT.

# (Optional) If fastp merging is not sufficient, or you need more control,
# use pRESTO's AssemblePairs after primer masking.
# For now, let's assume we will process the QC'd R1_QC and R2_QC further if needed,
# especially for primer trimming if primers are internal and not adapters.

# 2. Mask Primers (if primers are not fully handled by fastp's adapter trimming)
# This step helps get clean V(D)J sequences.
echo "Step 2: Masking primers (if applicable)..."
# For R1 (forward primers)
MaskPrimers.py align \
    -s "$R1_QC" \
    -p "../$FWD_PRIMER_FASTA" \
    --mode cut \
    --start 0 \
    --pf VPRIMER \
    --outname "${SAMPLE_NAME}_R1.mp" \
    --log MaskPrimers_R1.log \
    --nproc "$THREADS"

# For R2 (reverse primers)
MaskPrimers.py align \
    -s "$R2_QC" \
    -p "../$REV_PRIMER_FASTA" \
    --mode cut \
    --start 0 \
    --pf CPRIMER \
    --outname "${SAMPLE_NAME}_R2.mp" \
    --log MaskPrimers_R2.log \
    --nproc "$THREADS"

R1_PRIMER_TRIMMED="${SAMPLE_NAME}_R1.mp_primers-pass.fastq"
R2_PRIMER_TRIMMED="${SAMPLE_NAME}_R2.mp_primers-pass.fastq"

if [ ! -s "$R1_PRIMER_TRIMMED" ] || [ ! -s "$R2_PRIMER_TRIMMED" ]; then
    echo "Warning: Primer trimming resulted in empty files for R1 or R2. This might be okay if fastp already merged/handled them, or primers were not found. Check logs."
    # Don't exit if fastp already merged many, but this is a point to check.
fi


# 3. Assemble Paired Reads (if not relying solely on fastp merging or if primers were trimmed after fastp)
# This uses pRESTO's AssemblePairs, which is robust.
# It will try to merge reads from R1_PRIMER_TRIMMED and R2_PRIMER_TRIMMED.
# If fastp already did a good job merging, many reads might be single-end here.
# AssemblePairs can handle a mix or just one input if the other is empty.

echo "Step 3: Assembling paired reads (using pRESTO AssemblePairs)..."
ASSEMBLED_READS_PRESTO="${SAMPLE_NAME}.assembled_presto-pass.fastq"

# We need to handle the case where primer trimming might make one file empty
# or if fastp merged most things.
# A more robust way is to check if files exist and have content.
# For simplicity, we assume both $R1_PRIMER_TRIMMED and $R2_PRIMER_TRIMMED are valid inputs for AssemblePairs.
# If one is empty, AssemblePairs might complain or produce empty output.
if [ -s "$R1_PRIMER_TRIMMED" ] && [ -s "$R2_PRIMER_TRIMMED" ]; then
    AssemblePairs.py align \
        -1 "$R1_PRIMER_TRIMMED" \
        -2 "$R2_PRIMER_TRIMMED" \
        --coord presto \
        --rc tail \
        --outname "${SAMPLE_NAME}.assembled_presto" \
        --log AssemblePairs.log \
        --nproc "$THREADS"
else
    echo "Skipping pRESTO AssemblePairs as one or both primer-trimmed inputs are empty. Relying on fastp merge."
    # If fastp merged, $MERGED_FASTP_OUT is key. If not, this pipeline needs adjustment.
    # For this "simple" script, we'll assume $MERGED_FASTP_OUT is the primary if pRESTO assembly isn't run.
fi

# Determine which merged file to use for the next step
MERGED_FOR_FILTERING=""
if [ -s "$ASSEMBLED_READS_PRESTO" ]; then
    MERGED_FOR_FILTERING="$ASSEMBLED_READS_PRESTO"
    echo "Using pRESTO assembled reads for filtering."
elif [ -s "$MERGED_FASTP_OUT" ]; then
    MERGED_FOR_FILTERING="$MERGED_FASTP_OUT"
    echo "Using fastp merged reads for filtering."
else
    echo "ERROR: No merged reads available from fastp or pRESTO. Check logs."
    exit 1
fi

# 4. Final Filtering of Merged Sequences (Example: by length) and Convert to FASTA
echo "Step 4: Filtering merged sequences and converting to FASTA..."
FINAL_IGBLAST_INPUT_FASTA="${SAMPLE_NAME}.final_merged.fasta" # This is your IgBLAST input

FilterSeq.py length \
    -s "$MERGED_FOR_FILTERING" \
    -n 200 \
    --fasta \
    --outname "${SAMPLE_NAME}.final_merged" \
    --log FilterSeq_final.log
    # -n 200 means minimum length 200bp. Adjust as needed.

echo "NON-UMI preprocessing complete!"
echo "Final merged FASTA for IgBLAST: $OUT_DIR/$FINAL_IGBLAST_INPUT_FASTA"
echo "Log files are in $OUT_DIR"
cd ..
