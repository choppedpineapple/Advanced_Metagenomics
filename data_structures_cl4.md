In bash, FASTQ files present unique challenges because they're structured text with a specific 4-line pattern, but bash isn't naturally designed for complex data structures. However, there are several effective approaches you can use:

## Arrays - Your Primary Tool

**Associative Arrays** are like filing cabinets with labeled drawers. Each sequence gets its own compartment:

```bash
#!/bin/bash
declare -A sequences
declare -A qualities

# Reading FASTQ into associative arrays
while IFS= read -r header && IFS= read -r seq && IFS= read -r plus && IFS= read -r qual; do
    # Extract sequence ID (remove @ and everything after first space)
    seq_id=${header#@}
    seq_id=${seq_id%% *}
    
    sequences["$seq_id"]="$seq"
    qualities["$seq_id"]="$qual"
done < input.fastq

# Access any sequence by ID
echo "Sequence for SRR123456.1: ${sequences[SRR123456.1]}"
```

**Indexed Arrays** work like numbered parking spaces - perfect for processing FASTQ records in order:

```bash
#!/bin/bash
declare -a headers
declare -a sequences
declare -a qualities

index=0
while IFS= read -r header && IFS= read -r seq && IFS= read -r plus && IFS= read -r qual; do
    headers[$index]="$header"
    sequences[$index]="$seq"
    qualities[$index]="$qual"
    ((index++))
done < input.fastq

# Process first 10 sequences
for ((i=0; i<10; i++)); do
    echo "Record $i: ${headers[$i]}"
    echo "Length: ${#sequences[$i]}"
done
```

## Stream Processing - The Assembly Line Approach

Think of this like a factory assembly line where each FASTQ record gets processed as it comes through:

```bash
#!/bin/bash
process_fastq_stream() {
    local count=0
    local total_length=0
    
    while IFS= read -r header && IFS= read -r seq && IFS= read -r plus && IFS= read -r qual; do
        # Process each record immediately
        seq_length=${#seq}
        total_length=$((total_length + seq_length))
        count=$((count + 1))
        
        # Filter sequences by length (example: keep only reads >50bp)
        if [ "$seq_length" -gt 50 ]; then
            echo "$header"
            echo "$seq"
            echo "$plus"
            echo "$qual"
        fi
    done < "$1"
    
    echo "Processed $count sequences, average length: $((total_length/count))" >&2
}

process_fastq_stream input.fastq > filtered.fastq
```

## File-Based Data Structures - The Library System

Sometimes it's better to organize data into separate files, like books in different sections of a library:

```bash
#!/bin/bash
# Split FASTQ by quality score ranges
mkdir -p high_quality medium_quality low_quality

while IFS= read -r header && IFS= read -r seq && IFS= read -r plus && IFS= read -r qual; do
    # Calculate average quality (simplified)
    avg_qual=$(echo "$qual" | od -An -tu1 | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i-33; print sum/NF}')
    
    if (( $(echo "$avg_qual > 30" | bc -l) )); then
        {
            echo "$header"
            echo "$seq"
            echo "$plus"
            echo "$qual"
        } >> high_quality/reads.fastq
    elif (( $(echo "$avg_qual > 20" | bc -l) )); then
        {
            echo "$header"
            echo "$seq"
            echo "$plus"
            echo "$qual"
        } >> medium_quality/reads.fastq
    else
        {
            echo "$header"
            echo "$seq"
            echo "$plus"
            echo "$qual"
        } >> low_quality/reads.fastq
    fi
done < input.fastq
```

## Temporary Files as Buffers - The Staging Area

Like a staging area in a warehouse, temporary files help you reorganize data:

```bash
#!/bin/bash
# Create temporary files for processing
temp_dir=$(mktemp -d)
trap "rm -rf $temp_dir" EXIT

# Extract just sequences for analysis
extract_sequences() {
    awk 'NR%4==2' "$1" > "$temp_dir/sequences.txt"
}

# Extract just quality scores
extract_qualities() {
    awk 'NR%4==0' "$1" > "$temp_dir/qualities.txt"
}

# Extract headers
extract_headers() {
    awk 'NR%4==1' "$1" > "$temp_dir/headers.txt"
}

# Process files
extract_sequences input.fastq
extract_qualities input.fastq
extract_headers input.fastq

# Now you can process each component separately
echo "Sequence count: $(wc -l < "$temp_dir/sequences.txt")"
echo "Average sequence length: $(awk '{total+=length($0)} END {print total/NR}' "$temp_dir/sequences.txt")"
```

## Hash Tables for Fast Lookups - The Phone Book

For tasks requiring fast lookups (like finding duplicates), use associative arrays as hash tables:

```bash
#!/bin/bash
declare -A seen_sequences
duplicates=0
unique=0

while IFS= read -r header && IFS= read -r seq && IFS= read -r plus && IFS= read -r qual; do
    if [[ -n "${seen_sequences[$seq]}" ]]; then
        echo "Duplicate found: $header" >&2
        ((duplicates++))
    else
        seen_sequences["$seq"]=1
        # Output unique sequences
        echo "$header"
        echo "$seq"
        echo "$plus"
        echo "$qual"
        ((unique++))
    fi
done < input.fastq > unique.fastq

echo "Found $duplicates duplicates, kept $unique unique sequences" >&2
```

## Pipeline Data Structures - The Conveyor Belt

Bash excels at pipelines, so structure your data flow like a conveyor belt:

```bash
#!/bin/bash
# Multi-stage processing pipeline
cat input.fastq | \
# Stage 1: Filter by length
awk '
{
    if (NR%4==1) header=$0
    else if (NR%4==2) {
        seq=$0
        if (length(seq) >= 50) {
            print header
            print seq
            getline plus
            print plus
            getline qual
            print qual
        } else {
            getline plus
            getline qual
        }
    }
}' | \
# Stage 2: Convert to FASTA for downstream tools
awk '
NR%4==1 {gsub(/^@/, ">"); print}
NR%4==2 {print}
NR%4==3 || NR%4==0 {next}
' > filtered.fasta
```

## Memory-Efficient Chunking - The Batch Processor

For large FASTQ files, process in chunks like a batch processor:

```bash
#!/bin/bash
process_fastq_chunks() {
    local input_file="$1"
    local chunk_size=10000  # Process 10k reads at a time
    local chunk_num=0
    
    while true; do
        # Read chunk_size records
        temp_chunk=$(mktemp)
        records_read=0
        
        while [ $records_read -lt $chunk_size ] && \
              IFS= read -r header && IFS= read -r seq && IFS= read -r plus && IFS= read -r qual; do
            {
                echo "$header"
                echo "$seq"
                echo "$plus"
                echo "$qual"
            } >> "$temp_chunk"
            ((records_read++))
        done < <(tail -n +$((chunk_num * chunk_size * 4 + 1)) "$input_file")
        
        if [ $records_read -eq 0 ]; then
            rm "$temp_chunk"
            break
        fi
        
        # Process this chunk
        echo "Processing chunk $chunk_num with $records_read records"
        process_chunk "$temp_chunk"
        
        rm "$temp_chunk"
        ((chunk_num++))
    done
}

process_chunk() {
    local chunk_file="$1"
    # Your processing logic here
    grep -c "^@" "$chunk_file"
}
```

The key insight is that bash isn't meant for complex in-memory data structures like Python or R. Instead, leverage bash's strengths: text processing, pipelines, and file manipulation. Think of your FASTQ processing like managing a library - sometimes you need quick reference (associative arrays), sometimes you need to process everything in order (streams), and sometimes you need to reorganize into different collections (file-based structures).

For truly complex bioinformatics tasks, consider calling specialized tools from bash rather than implementing everything in pure bash - bash excels at orchestrating tools rather than replacing them.
