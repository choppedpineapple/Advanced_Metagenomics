#!/usr/bin/bash

# Color definitions for better output readability
NC='\033[0m'         # Text Reset
IYellow='\033[0;93m' # Yellow
IGreen='\033[0;92m'  # Green
IRed='\033[0;91m'    # Red
BBlue='\033[1;34m'   # Bold Blue

### Command line arguments ###
pd=${1}             # Where the executable files are saved
work_dir=${2}       # Working directory
prj=${3}            # Project / main folder with the fastq
threads=${4}        # Number of threads
timestamp=${5}      # Timestamp for logging and output organization

# Create organized output directories structure
output_dir="${pd}/output"
fastqc_dir="${output_dir}/fastqc"
merged_dir="${output_dir}/merged"
result_dir="${output_dir}/result"
igblast_dir="${output_dir}/igblast"
logs_dir="${output_dir}/logs"

# Start time tracking
start_time=$(date +%s)

# Input directory
input_dir="${work_dir}/${prj}"

# Log function to standardize output
log() {
  local level=$1
  local message=$2
  local color=${NC}
  
  case $level in
    "INFO")  color=${IGreen};;
    "ERROR") color=${IRed};;
    "STEP")  color=${BBlue};;
  esac
  
  echo -e "\n${color}[$level] $(date '+%Y-%m-%d %H:%M:%S') - $message${NC}" | tee -a "${logs_dir}/pipeline_${timestamp}.log"
}

# Error handler function
handle_error() {
  log "ERROR" "Step $1 failed with exit code $2"
  log "ERROR" "Pipeline aborted"
  exit $2
}

# Record start of pipeline
log "INFO" "Starting Alpaca Antibody Phage Sample Processing Pipeline"
log "INFO" "Input directory: ${input_dir}"
log "INFO" "Using ${threads} threads"

# Get sample ID from the first fastq file
cd ${input_dir}
sample=$(ls *.fastq.gz | head -n 1 | awk -F "_" '{print $1}')
cd ${pd}

log "INFO" "Processing sample: ${sample}"

### STEP 1 - FastQC Quality Control ###
log "STEP" "STEP 1: Running FastQC for quality control"

# Run FastQC with more threads for faster processing
fastqc ${input_dir}/*.fastq.gz -t ${threads} -o ${fastqc_dir} || handle_error "1 (FastQC)" $?

log "INFO" "FastQC completed successfully"

### STEP 2 - Merging paired-end reads using pandaseq ###
log "STEP" "STEP 2: Merging paired-end reads with pandaseq"

# Load conda environment
eval "$(conda shell.bash hook)"
conda activate igblast

# Use more threads for pandaseq and direct output to a specific directory
merged_file="${merged_dir}/${sample}_merged.fastq"
pandaseq -F -f ${input_dir}/*_R1_*.fastq.gz -r ${input_dir}/*_R2_*.fastq.gz \
  -l 400 -o 30 -T ${threads} -w ${merged_file} || handle_error "2 (pandaseq)" $?

conda deactivate
log "INFO" "Pandaseq merging completed successfully"

### STEP 3 - Sorting merged reads ###
log "STEP" "STEP 3: Sorting merged reads"
sorted_file="${result_dir}/sorted_file.txt"
python ${pd}/sorted.py "${sorted_file}" "${merged_file}" || handle_error "3 (sorting)" $?
log "INFO" "Sorting completed successfully"

### STEP 4 - Splitting files ###
log "STEP" "STEP 4: Splitting files"
python ${pd}/split_file.py "${result_dir}" "${sorted_file}" || handle_error "4 (splitting)" $?
log "INFO" "File splitting completed successfully"

### STEP 5 - Consensus calling ###
log "STEP" "STEP 5: Consensus calling"
python ${pd}/consensus_calling_nonumi.py "${result_dir}" || handle_error "5 (consensus)" $?
log "INFO" "Consensus calling completed successfully"

# Prepare IgBLAST environment
log "INFO" "Setting up IgBLAST environment"

# Extract IgBLAST files for alpaca and create necessary directories
unzip -o ${pd}/igblast_files_alpaca.zip -d ${igblast_dir} || handle_error "Unzipping IgBLAST files" $?

# Copy consensus file to the appropriate directory
mkdir -p "${igblast_dir}/igblast_files_alpaca/consensus_folder"
cp ${pd}/consensus_file.txt ${igblast_dir}/igblast_files_alpaca/consensus_folder/
cp ${pd}/IgBlast.py ${pd}/igblast_part2.py ${igblast_dir}/igblast_files_alpaca/

### STEP 6 - IgBLAST preprocessing ###
log "STEP" "STEP 6: IgBLAST preprocessing"
consensus_file="${igblast_dir}/igblast_files_alpaca/consensus_folder/consensus_file.txt"
python ${pd}/igblast_pre.py "${consensus_file}" || handle_error "6 (IgBLAST preprocessing)" $?
log "INFO" "IgBLAST preprocessing completed successfully"

# Copy required files and set working directory for IgBLAST
cp ${pd}/SEQ_COUNT_AS_HEADER.txt ${igblast_dir}/igblast_files_alpaca/
cd ${igblast_dir}/igblast_files_alpaca/

### STEP 7 - Running IgBLAST ###
log "STEP" "STEP 7: Running IgBLAST"
igblast_output="${igblast_dir}/igblast_files_alpaca/igblast_output.tsv"

eval "$(conda shell.bash hook)"
conda activate igblast
python3 ${PWD}/IgBlast.py "${igblast_output}" || handle_error "7 (IgBLAST)" $?
conda deactivate
log "INFO" "IgBLAST analysis completed successfully"

### STEP 8 - IgBLAST post-processing ###
log "STEP" "STEP 8: IgBLAST post-processing"
python3 ${PWD}/igblast_part2.py "${igblast_output}" || handle_error "8 (IgBLAST post-processing)" $?
log "INFO" "IgBLAST post-processing completed successfully"

# Return to original directory and organize output files
cd ${pd}

# Calculate elapsed time
end_time=$(date +%s)
elapsed_seconds=$((end_time - start_time))
hours=$((elapsed_seconds / 3600))
minutes=$(( (elapsed_seconds % 3600) / 60 ))
seconds=$((elapsed_seconds % 60))

# Print completion message with formatted time
log "INFO" "Pipeline completed successfully"
echo -e "\n${IGreen}#############################################################${NC}"
echo -e "${IGreen}<<< Alpaca Antibody Phage Sample Processing Pipeline Completed >>>${NC}"
echo -e "${IGreen}Pipeline completed in ${hours} hours, ${minutes} minutes, ${seconds} seconds${NC}"
echo -e "${IGreen}Output files are organized in: ${output_dir}${NC}"
echo -e "${IGreen}#############################################################${NC}\n"

exit 0
