import gzip
from Bio import SeqIO

def parse_with_biopython(filename):
    """
    Parse FASTQ file using Biopython's SeqIO module
    Returns generator of (header, sequence, quality) tuples
    """
    try:
        # Handle gzipped files transparently
        with gzip.open(filename, 'rt') if filename.endswith('.gz') else open(filename, 'r') as f:
            for record in SeqIO.parse(f, 'fastq'):
                yield (record.id, str(record.seq), record.letter_annotations['phred_quality'])
    except Exception as e:
        print(f"Biopython parsing error: {e}")

def parse_with_native(filename):
    """
    Parse FASTQ file using native Python implementation
    Returns generator of (header, sequence, quality) tuples
    """
    try:
        opener = gzip.open if filename.endswith('.gz') else open
        with opener(filename, 'rt') as f:
            while True:
                header = f.readline().strip()
                if not header:
                    break  # End of file
                
                sequence = f.readline().strip()
                separator = f.readline().strip()
                quality = f.readline().strip()

                # Basic validation
                if not header.startswith('@'):
                    raise ValueError(f"Invalid header: {header}")
                if not separator.startswith('+'):
                    raise ValueError(f"Invalid separator: {separator}")
                if len(sequence) != len(quality):
                    raise ValueError("Sequence/quality length mismatch")

                yield (header[1:], sequence, [ord(c) - 33 for c in quality])
                
    except Exception as e:
        print(f"Native parsing error: {e}")

def compare_parsers(filename):
    """
    Compare outputs and performance of both parsers
    """
    from time import time
    
    # Test Biopython parser
    start = time()
    bio_count = sum(1 for _ in parse_with_biopython(filename))
    bio_time = time() - start
    
    # Test Native parser
    start = time()
    native_count = sum(1 for _ in parse_with_native(filename))
    native_time = time() - start
    
    print(f"\nBiopython parsed {bio_count} records in {bio_time:.4f}s")
    print(f"Native parser parsed {native_count} records in {native_time:.4f}s")
    print("Records match" if bio_count == native_count else "Record count mismatch!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python fastq_parser.py <input.fastq[.gz]>")
        sys.exit(1)
    
    compare_parsers(sys.argv[1])
