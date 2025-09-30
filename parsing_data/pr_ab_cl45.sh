#!/usr/bin/bash

##############################################################################
# VDJ Heavy Chain Assembly Pipeline for Sheep Antibody Data
# Illumina MiSeq 2x300 paired-end sequencing
# Tool: pRESTO
##############################################################################

# Set parameters
THREADS=16
OUTDIR="output_presto"
R1="ES-RAGE-heavy-S4_L001_R1_001.fastq.gz"
R2="ES-RAGE-heavy-S4_L001_R2_001.fastq.gz"
SAMPLE="ES-RAGE-heavy-S4"

# Quality thresholds
QUAL_THRESHOLD=20
MIN_LEN=250
MAX_ERROR=0.1

# IgBLAST parameters (adjust paths as needed)
IGBLAST_DB="./igblast_db"
GERMLINE_DB_V="${IGBLAST_DB}/sheep_V"
GERMLINE_DB_D="${IGBLAST_DB}/sheep_D"
GERMLINE_DB_J="${IGBLAST_DB}/sheep_J"
AUXILIARY="${IGBLAST_DB}/optional_file"

# Create output directory
mkdir -p ${OUTDIR}
cd ${OUTDIR}

echo "=========================================="
echo "Starting VDJ Heavy Chain Assembly Pipeline"
echo "Sample: ${SAMPLE}"
echo "Threads: ${THREADS}"
echo "=========================================="

##############################################################################
# STEP 1: Quality Control and Filtering
##############################################################################
echo ""
echo "STEP 1: Quality filtering of reads..."

# Filter R1 by quality
FilterSeq.py quality \
    -s ../${R1} \
    -q ${QUAL_THRESHOLD} \
    --nproc ${THREADS} \
    --outname ${SAMPLE}_R1 \
    --outdir . \
    --log FS1.log

# Filter R2 by quality
FilterSeq.py quality \
    -s ../${R2} \
    -q ${QUAL_THRESHOLD} \
    --nproc ${THREADS} \
    --outname ${SAMPLE}_R2 \
    --outdir . \
    --log FS2.log

echo "Quality filtering complete."

##############################################################################
# STEP 2: Paired-end Assembly
##############################################################################
echo ""
echo "STEP 2: Assembling paired-end reads..."

# Assemble paired ends using reference assembly mode
# This will align R1 and R2 and create consensus sequences
AssemblePairs.py align \
    -1 ${SAMPLE}_R1_quality-pass.fastq \
    -2 ${SAMPLE}_R2_quality-pass.fastq \
    --coord illumina \
    --rc tail \
    --maxerror ${MAX_ERROR} \
    --minlen ${MIN_LEN} \
    --maxlen 600 \
    --alpha 1e-5 \
    --scanrev \
    --nproc ${THREADS} \
    --outname ${SAMPLE} \
    --outdir . \
    --log AP.log

echo "Paired-end assembly complete."

##############################################################################
# STEP 3: Filter by Length and Quality
##############################################################################
echo ""
echo "STEP 3: Filtering assembled sequences by length..."

# Filter sequences by length (VH sequences typically 300-400bp)
FilterSeq.py length \
    -s ${SAMPLE}_assemble-pass.fastq \
    -n ${MIN_LEN} \
    --nproc ${THREADS} \
    --outname ${SAMPLE}_assembled \
    --outdir . \
    --log FS3.log

echo "Length filtering complete."

##############################################################################
# STEP 4: Remove Duplicates
##############################################################################
echo ""
echo "STEP 4: Removing duplicate sequences..."

# Since no UMIs are available, collapse identical sequences
# This reduces computational burden for downstream analysis
CollapseSeq.py \
    -s ${SAMPLE}_assembled_length-pass.fastq \
    -n 0 \
    --uf CONSCOUNT \
    --cf CONSCOUNT \
    --act sum \
    --inner \
    --nproc ${THREADS} \
    --outname ${SAMPLE}_collapsed \
    --outdir . \
    --log CS.log

echo "Duplicate removal complete."

##############################################################################
# STEP 5: Convert to FASTA for IgBLAST
##############################################################################
echo ""
echo "STEP 5: Converting to FASTA format..."

# Convert final FASTQ to FASTA for IgBLAST analysis
seqkit fq2fa ${SAMPLE}_collapsed_collapse-unique.fastq \
    -o ${SAMPLE}_final.fasta \
    -j ${THREADS}

echo "Conversion to FASTA complete."

##############################################################################
# STEP 6: IgBLAST Annotation
##############################################################################
echo ""
echo "STEP 6: Running IgBLAST annotation..."

# Run IgBLAST for V(D)J annotation
# Note: Adjust organism and auxiliary data file based on your setup
igblastn \
    -germline_db_V ${GERMLINE_DB_V} \
    -germline_db_D ${GERMLINE_DB_D} \
    -germline_db_J ${GERMLINE_DB_J} \
    -auxiliary_data ${AUXILIARY} \
    -domain_system imgt \
    -ig_seqtype Ig \
    -organism sheep \
    -outfmt '7 std qseq sseq btop' \
    -query ${SAMPLE}_final.fasta \
    -out ${SAMPLE}_igblast.txt \
    -num_threads ${THREADS}

echo "IgBLAST annotation complete."

##############################################################################
# STEP 7: Parse IgBLAST Results (Optional - using Change-O if available)
##############################################################################
echo ""
echo "STEP 7: Parsing IgBLAST results..."

# If Change-O is installed, parse IgBLAST output to tabular format
if command -v MakeDb.py &> /dev/null; then
    MakeDb.py igblast \
        -i ${SAMPLE}_igblast.txt \
        -s ${SAMPLE}_final.fasta \
        -r ${GERMLINE_DB_V}.fasta ${GERMLINE_DB_D}.fasta ${GERMLINE_DB_J}.fasta \
        --extended \
        --format airr \
        -o ${SAMPLE}_db-pass.tsv
    
    echo "IgBLAST results parsed to AIRR format."
else
    echo "Change-O not found. IgBLAST results saved as ${SAMPLE}_igblast.txt"
fi

##############################################################################
# STEP 8: Generate Summary Statistics
##############################################################################
echo ""
echo "STEP 8: Generating summary statistics..."

# Count sequences at each step
echo "=== Pipeline Summary ===" > ${SAMPLE}_summary.txt
echo "" >> ${SAMPLE}_summary.txt
echo "R1 quality-pass: $(grep -c "^@" ${SAMPLE}_R1_quality-pass.fastq)" >> ${SAMPLE}_summary.txt
echo "R2 quality-pass: $(grep -c "^@" ${SAMPLE}_R2_quality-pass.fastq)" >> ${SAMPLE}_summary.txt
echo "Assembled pairs: $(grep -c "^@" ${SAMPLE}_assemble-pass.fastq)" >> ${SAMPLE}_summary.txt
echo "Length-filtered: $(grep -c "^@" ${SAMPLE}_assembled_length-pass.fastq)" >> ${SAMPLE}_summary.txt
echo "Unique sequences: $(grep -c "^>" ${SAMPLE}_final.fasta)" >> ${SAMPLE}_summary.txt
echo "" >> ${SAMPLE}_summary.txt

# Parse logs for additional statistics
echo "=== Assembly Statistics ===" >> ${SAMPLE}_summary.txt
grep "PASS" AP.log | tail -1 >> ${SAMPLE}_summary.txt
echo "" >> ${SAMPLE}_summary.txt

cat ${SAMPLE}_summary.txt

##############################################################################
# STEP 9: Clean Up Intermediate Files (Optional)
##############################################################################
echo ""
echo "Pipeline complete!"
echo "=========================================="
echo "Final outputs:"
echo "  - Assembled sequences: ${SAMPLE}_final.fasta"
echo "  - IgBLAST results: ${SAMPLE}_igblast.txt"
if [ -f "${SAMPLE}_db-pass.tsv" ]; then
    echo "  - Parsed database: ${SAMPLE}_db-pass.tsv"
fi
echo "  - Summary statistics: ${SAMPLE}_summary.txt"
echo "=========================================="

# Uncomment to remove intermediate files
# rm *quality-pass.fastq *assemble-fail.fastq *length-fail.fastq

cd ..
echo "All done!"
