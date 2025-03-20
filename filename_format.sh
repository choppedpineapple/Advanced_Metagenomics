#!/bin/bash

# Check if files in sample directory match Illumina naming convention
illumina_pattern="_S[0-9]+_L00[0-9]_R[12]_001\.fastq\.gz$"

# Count the number of fastq.gz files
fastq_count=$(find "${sample_input}" -maxdepth 1 -type f -name "*.fastq.gz" | wc -l)

# If no fastq.gz files found, exit
if [ "$fastq_count" -eq 0 ]; then
    echo "Error: No fastq.gz files found in ${sample_input}"
    exit 1
fi

# Check if each fastq.gz file follows the Illumina naming convention
invalid_files=0
for file in "${sample_input}"/*.fastq.gz; do
    if [[ ! $(basename "$file") =~ ${illumina_pattern} ]]; then
        echo "Error: $(basename "$file") does not follow Illumina naming convention"
        invalid_files=$((invalid_files + 1))
    fi
done

# Check if we have both R1 and R2 reads for paired-end data
r1_count=$(find "${sample_input}" -maxdepth 1 -type f -name "*_R1_001.fastq.gz" | wc -l)
r2_count=$(find "${sample_input}" -maxdepth 1 -type f -name "*_R2_001.fastq.gz" | wc -l)

if [ "$r1_count" -ne "$r2_count" ]; then
    echo "Error: Unequal number of R1 (${r1_count}) and R2 (${r2_count}) files found"
    exit 1
fi

# Exit if any invalid files were found
if [ "$invalid_files" -gt 0 ]; then
    echo "Error: ${invalid_files} files do not follow the Illumina naming convention"
    echo "Expected format: sample_S2_L001_R1_001.fastq.gz and sample_S2_L001_R2_001.fastq.gz"
    exit 1
fi

echo "All files follow Illumina naming convention"

```
This script:
1. Defines the Illumina pattern to match against
2. Counts the total number of fastq.gz files
3. Checks each file against the pattern
4. Verifies you have equal numbers of R1 and R2 files (for paired-end data)
5. Exits with an error message if any files don't match or if the R1/R2 counts are unequal

You can integrate this snippet into your larger workflow script.
```
