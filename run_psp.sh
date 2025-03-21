#!/usr/bin/bash

# Current directory where the scripts are located
pd=${PWD}

# Number of threads to use - increased for better performance
threads=64

# Set memory based on thread count
mem=$((threads * 2))G

# Colors for terminal output
NC='\033[0m'         # Text Reset
IYellow='\033[0;93m' # Yellow
IRed='\033[0;91m'    # Red
IGreen='\033[0;92m'  # Green

# Get work directory
echo -n -e "\n${IYellow}Enter the path to the working directory: ${NC}"
read work_dir

if [ ! -d "${work_dir}" ]; then
  echo -e "\n${IRed}Error: ${work_dir} does not exist.${NC}\n"
  exit 1
fi

# Get project directory name
echo -n -e "\n${IYellow}Enter the name of the project directory containing fastq files: ${NC}"
read prj

# Verify project directory exists
if [ ! -d "${work_dir}/${prj}" ]; then
  echo -e "\n${IRed}Error: ${prj} is not present inside working directory.${NC}\n"
  exit 1
fi

# Create output directories in advance
mkdir -p "${pd}/output/fastqc"
mkdir -p "${pd}/output/merged"
mkdir -p "${pd}/output/result"
mkdir -p "${pd}/output/igblast"
mkdir -p "${pd}/output/logs"

# Use a timestamp for job naming and logging
timestamp=$(date +%Y%m%d_%H%M%S)
job_name="alpaca_pipeline_${timestamp}"
log_file="${pd}/output/logs/slurm_log_${timestamp}.log"

# Running the pipeline with improved resource allocation
sbatch --job-name=${job_name} \
  --partition=high \
  --cpus-per-task=${threads} \
  --mem=${mem} \
  --time=3-00:00:00 \
  --output=${log_file} \
  ${pd}/phage-sample-processing.sh ${pd} ${work_dir} ${prj} ${threads} ${timestamp}

echo
echo -e "${IYellow}###################################################################${NC}"
echo -e "${IYellow}<<<<-----Alpaca Antibody Phage Sample Processing Pipeline Starting----->>>>${NC} ${IGreen} $(date '+%Y/%m/%d %H:%M:%S') ${NC}"
echo -e "${IYellow}Job name: ${job_name}${NC}"
echo -e "${IYellow}Log file: ${log_file}${NC}"
echo -e "${IYellow}###################################################################${NC}"
echo
