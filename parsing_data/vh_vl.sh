#!/bin/bash
set -euo pipefail  # Strict error handling

# ==============================
# CONFIGURATION
# ==============================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data"
OUT_DIR="$SCRIPT_DIR/output"
LOG_DIR="$SCRIPT_DIR/logs"
DB_DIR="$SCRIPT_DIR/sheep_igblast_db"

# Input files (update if your filenames differ)
HEAVY_R1="$DATA_DIR/heavy_R1.fastq.gz"
HEAVY_R2="$DATA_DIR/heavy_R2.fastq.gz"
LIGHT_R1="$DATA_DIR/light_R1.fastq.gz"
LIGHT_R2="$DATA_DIR/light_R2.fastq.gz"

# Output files
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Length filters (adjust based on your amplicon design)
MIN_LEN_H=300
MAX_LEN_H=550
MIN_LEN_L=300
MAX_LEN_L=500

# Linker (GGGGS x3, nucleotide)
LINKER_NT="GGTGGTGGTTCTGGTGGTGGTTCTGGTGGTGGTTCT"

# ==============================
# VALIDATION
# ==============================
log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" >&2; }

# Check inputs
for f in "$HEAVY_R1" "$HEAVY_R2" "$LIGHT_R1" "$LIGHT_R2"; do
  if [[ ! -f "$f" ]]; then
    log "ERROR: Missing input file: $f"
    exit 1
  fi
done

# Check IgBLAST DB
for gene in IGHV IGHD IGHJ IGKV IGKJ; do
  if [[ ! -f "$DB_DIR/${gene}.fasta" ]]; then
    log "WARNING: Missing germline file: $DB_DIR/${gene}.fasta"
    log "You MUST provide sheep V/D/J references. See IMGT or OAS."
    exit 1
  fi
done

# Check tools
for tool in AssemblePairs.py igblastn makeblastdb; do
  if ! command -v "$tool" &> /dev/null; then
    log "ERROR: Required tool not found: $tool"
    exit 1
  fi
done

log "✅ Validation passed. Starting pipeline..."

# ==============================
# STEP 1: ASSEMBLE HEAVY CHAIN
# ==============================
log "Step 1: Assembling heavy chain reads..."
AssemblePairs.py join \
  --1 "$HEAVY_R1" \
  --2 "$HEAVY_R2" \
  --rc tail \
  --coord sanger \
  --minlen "$MIN_LEN_H" \
  --maxlen "$MAX_LEN_H" \
  --outdir "$OUT_DIR/heavy_assembled" \
  --log "$LOG_DIR/heavy_assemble.log" || { log "Heavy assembly failed"; exit 1; }

HEAVY_FASTA="$OUT_DIR/heavy_assembled/assemble.fasta"
if [[ ! -s "$HEAVY_FASTA" ]]; then
  log "ERROR: Heavy assembly produced empty output"
  exit 1
fi
log "Heavy assembly: $(grep -c '^>' "$HEAVY_FASTA") sequences"

# ==============================
# STEP 2: ASSEMBLE LIGHT CHAIN
# ==============================
log "Step 2: Assembling light chain reads..."
AssemblePairs.py join \
  --1 "$LIGHT_R1" \
  --2 "$LIGHT_R2" \
  --rc tail \
  --coord sanger \
  --minlen "$MIN_LEN_L" \
  --maxlen "$MAX_LEN_L" \
  --outdir "$OUT_DIR/light_assembled" \
  --log "$LOG_DIR/light_assemble.log" || { log "Light assembly failed"; exit 1; }

LIGHT_FASTA="$OUT_DIR/light_assembled/assemble.fasta"
if [[ ! -s "$LIGHT_FASTA" ]]; then
  log "ERROR: Light assembly produced empty output"
  exit 1
fi
log "Light assembly: $(grep -c '^>' "$LIGHT_FASTA") sequences"

# ==============================
# STEP 3: ANNOTATE WITH IGBLAST
# ==============================
log "Step 3: Running IgBLAST on heavy chain..."
igblastn \
  -germline_db_V "$DB_DIR/IGHV" \
  -germline_db_D "$DB_DIR/IGHD" \
  -germline_db_J "$DB_DIR/IGHJ" \
  -domain_system imgt \
  -organism sheep \
  -ig_seqtype Ig \
  -num_alignments_V 1 \
  -num_alignments_D 1 \
  -num_alignments_J 1 \
  -outfmt 19 \
  -query "$HEAVY_FASTA" \
  -out "$OUT_DIR/heavy_igblast.tsv" || { log "IgBLAST heavy failed"; exit 1; }

log "Step 3: Running IgBLAST on light chain..."
igblastn \
  -germline_db_V "$DB_DIR/IGKV" \
  -germline_db_J "$DB_DIR/IGKJ" \
  -domain_system imgt \
  -organism sheep \
  -ig_seqtype Ig \
  -outfmt 19 \
  -query "$LIGHT_FASTA" \
  -out "$OUT_DIR/light_igblast.tsv" || { log "IgBLAST light failed"; exit 1; }

# ==============================
# STEP 4: BUILD scFv (Python)
# ==============================
log "Step 4: Building scFv constructs..."

# Create Python script on-the-fly
cat > "$OUT_DIR/build_scFv.py" <<EOF
#!/usr/bin/env python3
import pandas as pd
import sys
from Bio.Seq import Seq

def translate_clean(seq):
    seq = seq[:len(seq)//3*3]
    return str(Seq(seq).translate()).replace('*', 'X')

# Load data
try:
    heavy = pd.read_csv("$OUT_DIR/heavy_igblast.tsv", sep="\\t")
    light = pd.read_csv("$OUT_DIR/light_igblast.tsv", sep="\\t")
except Exception as e:
    print(f"Error loading IgBLAST TSV: {e}", file=sys.stderr)
    sys.exit(1)

# Filter productive, in-frame, no stop in V-region
def filter_productive(df):
    return df[
        (df['productive'] == True) &
        (df['vj_in_frame'] == True) &
        (df['stop_codon'] == False)
    ].copy()

heavy_prod = filter_productive(heavy)
light_prod = filter_productive(light)

if heavy_prod.empty or light_prod.empty:
    print("No productive sequences found!", file=sys.stderr)
    sys.exit(1)

# Get top N by sequence (or by count if 'duplicate_count' exists)
N = 100
vh_list = heavy_prod['sequence_alignment'].dropna().head(N).tolist()
vl_list = light_prod['sequence_alignment'].dropna().head(N).tolist()

linker_nt = "$LINKER_NT"

with open("$OUT_DIR/scFv_constructs_nt.fasta", "w") as f_nt, \\
     open("$OUT_DIR/scFv_constructs_aa.fasta", "w") as f_aa:
    count = 0
    for i, vh in enumerate(vh_list):
        for j, vl in enumerate(vl_list):
            if not vh or not vl:
                continue
            scfv_nt = str(vh) + linker_nt + str(vl)
            scfv_aa = translate_clean(scfv_nt)
            if len(scfv_aa) < 200:  # sanity check
                continue
            name = f"scFv_H{i}_L{j}"
            f_nt.write(f">{name}\\n{scfv_nt}\\n")
            f_aa.write(f">{name}\\n{scfv_aa}\\n")
            count += 1

print(f"✅ Built {count} scFv constructs", file=sys.stderr)
EOF

chmod +x "$OUT_DIR/build_scFv.py"
python3 "$OUT_DIR/build_scFv.py" || { log "scFv construction failed"; exit 1; }

# ==============================
# FINAL QC
# ==============================
NT_COUNT=$(grep -c '^>' "$OUT_DIR/scFv_constructs_nt.fasta" 2>/dev/null || echo 0)
AA_COUNT=$(grep -c '^>' "$OUT_DIR/scFv_constructs_aa.fasta" 2>/dev/null || echo 0)

if [[ $NT_COUNT -eq 0 || $AA_COUNT -eq 0 ]]; then
  log "ERROR: No scFv constructs generated!"
  exit 1
fi

log "🎉 Pipeline completed successfully!"
log "Output:"
log "  - Nucleotide scFv: $OUT_DIR/scFv_constructs_nt.fasta ($NT_COUNT sequences)"
log "  - Amino acid scFv: $OUT_DIR/scFv_constructs_aa.fasta ($AA_COUNT sequences)"
log "  - Full logs in: $LOG_DIR/"
