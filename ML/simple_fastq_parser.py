import time

FASTQ_PATH = "/home/abhi/workspace/1.datasets/16S_gut_microbiome/sample_1/SRR32461054_1.fastq"

def parse_fastq_sequences(file_path):
    """Parses a FASTQ file and yields the sequence for each read."""
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            # The sequence is on every 4th line, starting from the 2nd line (index 1)
            if i % 4 == 1:
                yield line.strip()

if __name__ == "__main__":
    print(f"Starting CPU parsing of {FASTQ_PATH}...")
    start_time = time.time()

    # Use a list comprehension to gather all sequences
    sequences = [seq for seq in parse_fastq_sequences(FASTQ_PATH)]

    end_time = time.time()

    print(f"\nParsing complete. Found {len(sequences):,} sequences.")
    print(f"CPU parsing took: {end_time - start_time:.4f} seconds.")

    print("\n--- First 5 sequences ---")
    for i in range(5):
        print(f"{i+1}: {sequences[i]}")
