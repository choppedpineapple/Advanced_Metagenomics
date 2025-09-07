import dnaio
import argparse
from typing import Optional

def trim_read_based_on_average_quality(record: dnaio.Sequence) -> Optional[dnaio.Sequence]:
    """
    Trim a read based on average Phred quality score.
    If the average quality of the read is below 20, trim from the end until the average quality 
    of the remaining sequence is at least 20. If the entire read fails, discard it.

    Args:
        record (dnaio.Sequence): A read record from dnaio.

    Returns:
        Optional[dnaio.Sequence]: Trimmed read or None if discarded.
    """
    qualities = record.qualities
    n = len(qualities)
    if n == 0:
        return None  # Discard empty reads

    # Convert quality string to list of Phred scores
    phred_scores = [ord(q) - 33 for q in qualities]
    total_sum = sum(phred_scores)

    # Check if the entire read already meets the quality threshold
    if total_sum / n >= 20:
        return record

    # Iterate from the end to find the longest prefix with average quality >= 20
    current_sum = total_sum
    k = n
    while k > 0:
        if current_sum / k >= 20:
            break
        # Subtract the quality score of the last base and reduce length
        current_sum -= phred_scores[k-1]
        k -= 1
    else:
        k = 0  # Discard the entire read if no segment meets the threshold

    if k == 0:
        return None
    elif k < n:
        return record[:k]  # Return trimmed read
    else:
        return record  # No trimming needed

def main(input_file: str, output_file: str):
    """
    Process the FASTQ file: read, trim, and write reads.

    Args:
        input_file (str): Path to input FASTQ file.
        output_file (str): Path to output FASTQ file.
    """
    with dnaio.open(input_file) as reader, dnaio.open(output_file, mode='w') as writer:
        for record in reader:
            trimmed_record = trim_read_based_on_average_quality(record)
            if trimmed_record is not None:
                writer.write(trimmed_record)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trim FASTQ reads based on average quality below 20.")
    parser.add_argument("input", help="Input FASTQ file")
    parser.add_argument("output", help="Output FASTQ file")
    args = parser.parse_args()

    main(args.input, args.output)
