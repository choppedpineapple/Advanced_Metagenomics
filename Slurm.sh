#!/bin/bash
set -euo pipefail  # Exit on error, undefined variable, or pipe failure

# Function to display usage
usage() {
  echo "Usage: $0 --work_dir <work_dir> --prj <project_name> --sequencer <miseq|novaseq|nextseq>"
  exit 1
}

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --work_dir) WORK_DIR="$2"; shift 2 ;;
    --prj) PRJ="$2"; shift 2 ;;
    --sequencer) SEQUENCER="$2"; shift 2 ;;
    *) usage ;;
  esac
done

# Validate inputs
if [[ -z "${WORK_DIR:-}" || -z "${PRJ:-}" || -z "${SEQUENCER:-}" ]]; then
  usage
fi

INPUT_DIR="${WORK_DIR}/${PRJ}"
OUTPUT_DIR="${WORK_DIR}/${PRJ}_output"

# Check if input directory exists
if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Error: Input directory ${INPUT_DIR} not found."
  exit 1
fi

# Validate sequencer type
VALID_SEQUENCERS=("miseq" "novaseq" "nextseq")
if [[ ! " ${VALID_SEQUENCERS[*]} " =~ " ${SEQUENCER} " ]]; then
  echo "Error: Invalid sequencer. Choose from: miseq, novaseq, nextseq."
  exit 1
fi

# Check conda environment
if ! conda env list | grep -q "humann_env"; then
  echo "Error: Conda environment 'humann_env' not found. Create it first."
  exit 1
fi

# Detect samples and validate sequencer-file consistency
echo "Detecting samples in ${INPUT_DIR}..."
mkdir -p "${OUTPUT_DIR}/merged_samples"

# Generate sample list and validate file counts
SAMPLE_ERROR=0
for SAMPLE_DIR in "${INPUT_DIR}"/*; do
  if [[ -d "${SAMPLE_DIR}" ]]; then
    SAMPLE=$(basename "${SAMPLE_DIR}")
    R1_FILES=("${SAMPLE_DIR}"/*{_R1,_1}.fastq.gz)
    R2_FILES=("${SAMPLE_DIR}"/*{_R2,_2}.fastq.gz)
    
    # Check sequencer vs file count (e.g., NextSeq expects 2 or 4 files)
    case "${SEQUENCER}" in
      "miseq")
        if [[ ${#R1_FILES[@]} -ne 1 || ${#R2_FILES[@]} -ne 1 ]]; then
          echo "Error: MiSeq sample ${SAMPLE} has ${#R1_FILES[@]} R1 files (expected 1)."
          SAMPLE_ERROR=1
        fi
        ;;
      "nextseq")
        if [[ ${#R1_FILES[@]} -lt 1 || ${#R2_FILES[@]} -lt 1 ]]; then
          echo "Error: NextSeq sample ${SAMPLE} has invalid R1/R2 files."
          SAMPLE_ERROR=1
        fi
        ;;
    esac

    # Merge lanes for NextSeq/NovaSeq
    if [[ "${SEQUENCER}" == "nextseq" || "${SEQUENCER}" == "novaseq" ]]; then
      cat "${R1_FILES[@]}" > "${OUTPUT_DIR}/merged_samples/${SAMPLE}_R1.fastq.gz"
      cat "${R2_FILES[@]}" > "${OUTPUT_DIR}/merged_samples/${SAMPLE}_R2.fastq.gz"
    else
      cp "${R1_FILES[0]}" "${OUTPUT_DIR}/merged_samples/${SAMPLE}_R1.fastq.gz"
      cp "${R2_FILES[0]}" "${OUTPUT_DIR}/merged_samples/${SAMPLE}_R2.fastq.gz"
    fi
  fi
done

# Exit if validation failed
if [[ "${SAMPLE_ERROR}" -eq 1 ]]; then
  echo "Validation failed. Fix input files or sequencer type."
  exit 1
fi

# Generate sample list for SLURM array
ls "${OUTPUT_DIR}/merged_samples"/*_R1.fastq.gz | sed 's/_R1.fastq.gz//' > "${OUTPUT_DIR}/sample_list.txt"
NUM_SAMPLES=$(wc -l < "${OUTPUT_DIR}/sample_list.txt")

# Submit SLURM job array
echo "Submitting SLURM job array for ${NUM_SAMPLES} samples..."
mkdir -p "${OUTPUT_DIR}/logs"
sbatch --job-name="${PRJ}_humann" \
  --array=1-${NUM_SAMPLES} \
  --output="${OUTPUT_DIR}/logs/%x_%A_%a.out" \
  --error="${OUTPUT_DIR}/logs/%x_%A_%a.err" \
  <<EOF
#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=batch

# Activate conda and run pipeline
conda activate humann_env
bash "$(dirname "$0")/humann_pipeline.slurm" \
  --sample "\$(sed -n \${SLURM_ARRAY_TASK_ID}p "${OUTPUT_DIR}/sample_list.txt")" \
  --output_dir "${OUTPUT_DIR}"
EOF

echo "Jobs submitted! Monitor logs in ${OUTPUT_DIR}/logs."
