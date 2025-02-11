#!/bin/bash

# Get input parameters
read -p "Enter the path to the working directory: " work_dir
read -p "Enter the project name: " prj

# Setup directories
input_dir="${work_dir}/${prj}"
output_dir="${work_dir}/${prj}_output"
logs_dir="${output_dir}/logs"

# Create necessary directories
mkdir -p "$output_dir" "$logs_dir"

# Create a samples array file
find "$input_dir" -maxdepth 1 -mindepth 1 -type d -printf '%f\n' > "${output_dir}/samples.txt"
num_samples=$(wc -l < "${output_dir}/samples.txt")

# Create config file
cat > "${output_dir}/config.sh" << EOF
WORK_DIR="${work_dir}"
PRJ="${prj}"
INPUT_DIR="${input_dir}"
OUTPUT_DIR="${output_dir}"
CONDA_ENV="humann3-biobakery"
EOF

# Submit SLURM job array
sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=humann3_${prj}
#SBATCH --array=1-${num_samples}
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=${logs_dir}/%A_%a.out
#SBATCH --error=${logs_dir}/%A_%a.err

# Load config
source "${output_dir}/config.sh"

# Get sample ID from array index
SAMPLE=\$(sed -n "\${SLURM_ARRAY_TASK_ID}p" "${output_dir}/samples.txt")

# Activate conda environment
source /path/to/conda/etc/profile.d/conda.sh
conda activate \$CONDA_ENV

# Execute main processing script
bash /path/to/humann3_process.sh "\$SAMPLE"
EOF
