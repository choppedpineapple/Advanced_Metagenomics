import gzip
import sys

def calculate_average_qscore(fastq_path):
    total_q_score = 0
    total_bases = 0
    
    # Handle both gzipped and raw files
    open_func = gzip.open if fastq_path.endswith('.gz') else open
    
    try:
        # 'rt' mode reads as text, not bytes
        with open_func(fastq_path, 'rt') as f:
            for i, line in enumerate(f):
                # FASTQ format: Quality scores are on every 4th line (index 3, 7, 11...)
                if (i + 1) % 4 == 0:
                    clean_line = line.strip()
                    
                    # Convert ASCII character to Phred score (offset 33)
                    # and add to our running total
                    total_q_score += sum(ord(char) - 33 for char in clean_line)
                    total_bases += len(clean_line)
                    
        if total_bases == 0:
            return 0.0
            
        return total_q_score / total_bases
        
    except FileNotFoundError:
        print("Yeah, that file doesn't exist. Try a real path.")
        sys.exit(1)

# Run it and face the music
fastq_file = "your_suspicious_reads.fq.gz" # Replace with your actual file
avg_q = calculate_average_qscore(fastq_file)

print(f"Total Bases: {total_bases}")
print(f"Calculated Average Quality Score: {avg_q:.2f}")
