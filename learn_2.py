# Script 2: filter_fastq_pandas.py

import argparse
import pandas as pd
import sys
from itertools import islice # Used for efficient 4-line reading

def filter_long_reads_pandas(input_fastq_files, output_file, min_length=100):
    """
    Reads FASTQ files, loads data into a pandas DataFrame, filters reads
    longer than min_length, and writes them back to the output file
    in FASTQ format.

    Args:
        input_fastq_files (list): A list of paths to input FASTQ files.
        output_file (str): Path to the output text file for long reads.
        min_length (int): Minimum sequence length threshold.
    """
    all_records = []
    count_total = 0
    count_long = 0

    print("Reading and parsing FASTQ files...")
    for fastq_file in input_fastq_files:
         print(f"Processing file: {fastq_file}...")
         try:
            with open(fastq_file, 'r') as infile:
                # Read 4 lines at a time
                while True:
                    lines = list(islice(infile, 4))
                    if not lines or len(lines) < 4:
                        break # End of file or incomplete record

                    count_total += 1
                    # Strip newline characters
                    record_id = lines[0].strip()
                    sequence = lines[1].strip()
                    plus = lines[2].strip()
                    quality = lines[3].strip()

                    # Basic validation (can be more robust)
                    if not record_id.startswith('@') or not plus.startswith('+'):
                         print(f"Warning: Skipping potentially malformed record starting near line {count_total * 4 - 3} in {fastq_file}", file=sys.stderr)
                         continue

                    all_records.append({
                        'id': record_id,
                        'sequence': sequence,
                        'plus': plus,
                        'quality': quality,
                        'length': len(sequence) # Calculate length here
                    })

         except FileNotFoundError:
             print(f"Error: Input file not found: {fastq_file}", file=sys.stderr)
             continue # Skip to next file if one is not found
         except Exception as e:
             print(f"Error reading file {fastq_file}: {e}", file=sys.stderr)
             continue # Skip to next file on error


    if not all_records:
        print("No valid records found or read.")
        return

    # Create DataFrame
    print("Creating DataFrame...")
    df = pd.DataFrame(all_records)

    # Filter DataFrame
    print(f"Filtering reads longer than {min_length} bps...")
    long_reads_df = df[df['length'] > min_length].copy() # Use .copy() to avoid SettingWithCopyWarning
    count_long = len(long_reads_df)

    # Write filtered reads back to FASTQ format
    print(f"Writing {count_long} long reads to {output_file}...")
    try:
        with open(output_file, 'w') as outfile:
            for index, row in long_reads_df.iterrows():
                outfile.write(f"{row['id']}\n")
                outfile.write(f"{row['sequence']}\n")
                outfile.write(f"{row['plus']}\n")
                outfile.write(f"{row['quality']}\n")

        print(f"\nProcessing complete.")
        print(f"Total reads processed: {count_total}")
        print(f"Reads longer than {min_length} bps: {count_long}")
        print(f"Long reads written to: {output_file}")

    except IOError as e:
        print(f"Error writing to output file {output_file}: {e}", file=sys.stderr)
    except Exception as e:
         print(f"An unexpected error occurred during writing: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Filter FASTQ reads longer than a specified length using pandas.")
    parser.add_argument("fastq_files",
                        metavar="FILE",
                        nargs='+', # Accept one or more files
                        help="Path(s) to input FASTQ file(s).")
    parser.add_argument("-o", "--output",
                        default="long_reads.txt",
                        help="Path to the output file (default: long_reads.txt)")
    parser.add_argument("-l", "--min_length",
                        type=int,
                        default=100,
                        help="Minimum read length to keep (default: 100)")

    args = parser.parse_args()

    filter_long_reads_pandas(args.fastq_files, args.output, args.min_length)
