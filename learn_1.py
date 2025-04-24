# Script 1: filter_fastq_biopython.py

import argparse
from Bio import SeqIO
import sys

def filter_long_reads_biopython(input_fastq_files, output_file, min_length=100):
    """
    Reads FASTQ files using BioPython, filters reads longer than min_length,
    and writes them to the output file in FASTQ format.

    Args:
        input_fastq_files (list): A list of paths to input FASTQ files.
        output_file (str): Path to the output text file for long reads.
        min_length (int): Minimum sequence length threshold.
    """
    count_total = 0
    count_long = 0

    try:
        with open(output_file, 'w') as outfile:
            for fastq_file in input_fastq_files:
                print(f"Processing file: {fastq_file}...")
                try:
                    # SeqIO.parse reads the file record by record (memory efficient)
                    for record in SeqIO.parse(fastq_file, "fastq"):
                        count_total += 1
                        if len(record.seq) > min_length:
                            count_long += 1
                            # Write the record in FASTQ format to the output file
                            SeqIO.write(record, outfile, "fastq")
                except FileNotFoundError:
                    print(f"Error: Input file not found: {fastq_file}", file=sys.stderr)
                except Exception as e:
                    print(f"Error processing file {fastq_file}: {e}", file=sys.stderr)

        print(f"\nProcessing complete.")
        print(f"Total reads processed: {count_total}")
        print(f"Reads longer than {min_length} bps: {count_long}")
        print(f"Long reads written to: {output_file}")

    except IOError as e:
        print(f"Error opening or writing to output file {output_file}: {e}", file=sys.stderr)
    except Exception as e:
         print(f"An unexpected error occurred: {e}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Filter FASTQ reads longer than a specified length using BioPython.")
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

    filter_long_reads_biopython(args.fastq_files, args.output, args.min_length)
