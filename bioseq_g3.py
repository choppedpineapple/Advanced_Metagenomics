import sys
import gzip
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

### File Reading Functions

def read_fastq(file_path):
    """
    Read a gzipped FASTQ file and yield SeqRecord objects.

    Args:
        file_path (str): Path to the gzipped FASTQ file.

    Yields:
        SeqRecord: Biopython SeqRecord object for each read.
    """
    with gzip.open(file_path, "rt") as handle:
        for record in SeqIO.parse(handle, "fastq"):
            yield record

def read_paired_end_fastq(file1, file2):
    """
    Read two gzipped FASTQ files for paired-end reads and yield pairs of SeqRecord objects.

    Args:
        file1 (str): Path to the first gzipped FASTQ file (e.g., _R1.fastq.gz).
        file2 (str): Path to the second gzipped FASTQ file (e.g., _R2.fastq.gz).

    Yields:
        tuple: Pair of SeqRecord objects (record1, record2).
    """
    with gzip.open(file1, "rt") as handle1, gzip.open(file2, "rt") as handle2:
        for record1, record2 in zip(SeqIO.parse(handle1, "fastq"), SeqIO.parse(handle2, "fastq")):
            yield record1, record2

### DNA Manipulation Function

def dna_string_manipulation(sequence):
    """
    Perform string manipulations on a DNA sequence: reverse complement and GC content.

    Args:
        sequence (str): DNA sequence as a string.

    Returns:
        tuple: (reverse complement as string, GC content as percentage).
    """
    seq = Seq(sequence.upper())
    rev_comp = seq.reverse_complement()
    gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
    return str(rev_comp), gc_content

### Quality Trimming Functions

def trim_quality(record, threshold=20):
    """
    Trim the sequence from the first base where quality falls below the threshold.

    Args:
        record (SeqRecord): Biopython SeqRecord object with quality scores.
        threshold (int): Quality score threshold (default is 20).

    Returns:
        SeqRecord: Trimmed SeqRecord object.
    """
    qualities = record.letter_annotations["phred_quality"]
    for i, q in enumerate(qualities):
        if q < threshold:
            return record[:i]
    return record

def sliding_window_trim(record, window_size=5, threshold=20):
    """
    Trim the sequence from the first window where average quality falls below the threshold.

    Args:
        record (SeqRecord): Biopython SeqRecord object with quality scores.
        window_size (int): Size of the sliding window (default is 5).
        threshold (int): Average quality threshold (default is 20).

    Returns:
        SeqRecord: Trimmed SeqRecord object.
    """
    qualities = record.letter_annotations["phred_quality"]
    seq_len = len(record)
    for i in range(seq_len - window_size + 1):
        window_qual = qualities[i:i + window_size]
        avg_qual = sum(window_qual) / window_size
        if avg_qual < threshold:
            return record[:i]
    return record

### Processing Functions

def process_single_end(file_path):
    """
    Process a single-end FASTQ file and print trimming results for the first 5 reads.

    Args:
        file_path (str): Path to the gzipped FASTQ file.
    """
    print(f"Processing single-end file: {file_path}")
    for i, record in enumerate(read_fastq(file_path)):
        if i >= 5:  # Limit to first 5 reads for demonstration
            break
        print(f"\nRead {i + 1}:")
        print(f"Original sequence: {record.seq}")
        print(f"Original qualities: {record.letter_annotations['phred_quality']}")
        
        trimmed_record = trim_quality(record)
        print(f"After quality trim: {trimmed_record.seq}")
        print(f"Trimmed qualities: {trimmed_record.letter_annotations['phred_quality']}")
        
        window_trimmed_record = sliding_window_trim(record)
        print(f"After sliding window trim: {window_trimmed_record.seq}")
        print(f"Window trimmed qualities: {window_trimmed_record.letter_annotations['phred_quality']}")

def process_paired_end(file1, file2):
    """
    Process paired-end FASTQ files and print trimming results for the first 5 read pairs.

    Args:
        file1 (str): Path to the first gzipped FASTQ file.
        file2 (str): Path to the second gzipped FASTQ file.
    """
    print(f"Processing paired-end files: {file1} and {file2}")
    for i, (record1, record2) in enumerate(read_paired_end_fastq(file1, file2)):
        if i >= 5:  # Limit to first 5 read pairs
            break
        print(f"\nRead pair {i + 1}:")
        
        print("Read 1:")
        print(f"Original sequence: {record1.seq}")
        print(f"Original qualities: {record1.letter_annotations['phred_quality']}")
        trimmed_record1 = trim_quality(record1)
        print(f"After quality trim: {trimmed_record1.seq}")
        print(f"Trimmed qualities: {trimmed_record1.letter_annotations['phred_quality']}")
        window_trimmed_record1 = sliding_window_trim(record1)
        print(f"After sliding window trim: {window_trimmed_record1.seq}")
        print(f"Window trimmed qualities: {window_trimmed_record1.letter_annotations['phred_quality']}")
        
        print("Read 2:")
        print(f"Original sequence: {record2.seq}")
        print(f"Original qualities: {record2.letter_annotations['phred_quality']}")
        trimmed_record2 = trim_quality(record2)
        print(f"After quality Vaughntrim: {trimmed_record2.seq}")
        print(f"Trimmed qualities: {trimmed_record2.letter_annotations['phred_quality']}")
        window_trimmed_record2 = sliding_window_trim(record2)
        print(f"After sliding window trim: {window_trimmed_record2.seq}")
        print(f"Window trimmed qualities: {window_trimmed_record2.letter_annotations['phred_quality']}")

### Main Function

def main():
    """
    Main function to parse command-line arguments and execute specific functions.

    Usage:
        python script.py single_end <file.fastq.gz>
        python script.py paired_end <file1.fastq.gz> <file2.fastq.gz>
        python script.py manipulate <sequence>
    """
    if len(sys.argv) < 2:
        print("Usage: python script.py <command> [args]")
        print("Commands: single_end, paired_end, manipulate")
        sys.exit(1)
    
    command = sys.argv[1]
    if command == "single_end":
        if len(sys.argv) != 3:
            print("Usage: python script.py single_end <file.fastq.gz>")
            sys.exit(1)
        file_path = sys.argv[2]
        process_single_end(file_path)
    elif command == "paired_end":
        if len(sys.argv) != 4:
            print("Usage: python script.py paired_end <file1.fastq.gz> <file2.fastq.gz>")
            sys.exit(1)
        file1 = sys.argv[2]
        file2 = sys.argv[3]
        process_paired_end(file1, file2)
    elif command == "manipulate":
        if len(sys.argv) != 3:
            print("Usage: python script.py manipulate <sequence>")
            sys.exit(1)
        sequence = sys.argv[2]
        rev_comp, gc_content = dna_string_manipulation(sequence)
        print(f"Original sequence: {sequence}")
        print(f"Reverse complement: {rev_comp}")
        print(f"GC content: {gc_content:.2f}%")
    else:
        print(f"Unknown command: {command}")
        print("Commands: single_end, paired_end, manipulate")

if __name__ == "__main__":
    main()
