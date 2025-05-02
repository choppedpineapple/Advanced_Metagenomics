import gzip
import io # Good practice, ensures text decoding

def parse_fastq(filename):
    """
    Parses a FASTQ file (plain '.fastq' or gzipped '.fastq.gz')
    and yields records one by one. This is memory-efficient.

    Args:
        filename (str): The path to the FASTQ file.

    Yields:
        tuple: A tuple containing (identifier_line, sequence, plus_line, quality_line)
               for each record in the file. All elements are strings.

    Raises:
        ValueError: If the file format appears corrupt (e.g., lines missing,
                    identifier doesn't start with '@', lengths mismatch).
        FileNotFoundError: If the input file cannot be found.
    """
    if not isinstance(filename, str) or not filename:
        raise ValueError("Invalid filename provided.")

    if filename.endswith(".gz"):
        try:
            # Use gzip.open in read text ('rt') mode with UTF-8 encoding (common standard)
            opener = gzip.open
            mode = 'rt'
            encoding = 'utf-8' # Explicitly state encoding
        except ImportError:
            raise ImportError("gzip module required to read .gz files.")
    else:
        # Use standard open in read text ('r') mode with UTF-8 encoding
        opener = open
        mode = 'r'
        encoding = 'utf-8'

    try:
        with opener(filename, mode, encoding=encoding) as f:
            record_count = 0
            while True:
                # 1. Read Identifier Line
                identifier = f.readline()
                if not identifier:  # End of file reached
                    break

                # Remove trailing newline/whitespace
                identifier = identifier.strip()

                # 2. Read Sequence Line
                sequence = f.readline().strip()

                # 3. Read Plus Line
                plus = f.readline().strip()

                # 4. Read Quality Line
                quality = f.readline().strip()

                record_count += 1 # Keep track for error messages

                # --- Basic Validation ---
                if not identifier.startswith('@'):
                    raise ValueError(f"Record #{record_count}: Identifier line does not start with '@'. Line content: '{identifier[:50]}...'")

                if not plus.startswith('+'):
                     raise ValueError(f"Record #{record_count}: Plus line does not start with '+'. Line content: '{plus[:50]}...'")

                if len(sequence) != len(quality):
                     raise ValueError(
                         f"Record #{record_count} (starting '{identifier[:30]}...'): "
                         f"Sequence length ({len(sequence)}) does not match quality length ({len(quality)})."
                     )

                # If all lines read and basic checks pass, yield the record
                yield (identifier, sequence, plus, quality)

    except FileNotFoundError:
        print(f"Error: File not found at '{filename}'")
        raise # Re-raise the exception
    except Exception as e:
        # Catch other potential errors (like decoding errors, ValueErrors from checks)
        print(f"An error occurred while parsing '{filename}': {e}")
        raise # Re-raise the exception

# --- Example Usage ---

# 1. Create dummy FASTQ files for testing
def create_dummy_files():
    # Plain FASTQ
    fastq_content = """@SEQ_ID_1 Read 1
GATTACA
+
<<<=<<<
@SEQ_ID_2 Read 2
AGCTAGCT
+
;;;;9;;;
@SEQ_ID_3 Read 3 with longer description
CATCATCATCAT
+SEQ_ID_3 Read 3 with longer description
FFFFFFFFFFFF
"""
    with open("example.fastq", "w", encoding='utf-8') as f:
        f.write(fastq_content)
    print("Created example.fastq")

    # Gzipped FASTQ
    with gzip.open("example.fastq.gz", "wt", encoding='utf-8') as f:
        f.write(fastq_content)
    print("Created example.fastq.gz")

create_dummy_files()

# 2. Process the plain FASTQ file
print("\n--- Processing example.fastq ---")
try:
    record_counter = 0
    for identifier, sequence, plus, quality in parse_fastq("example.fastq"):
        record_counter += 1
        print(f"Record {record_counter}:")
        print(f"  ID:   {identifier}")
        print(f"  Seq:  {sequence}")
        # print(f"  Plus: {plus}") # Usually not needed after parsing
        print(f"  Qual: {quality}")
        print("-" * 20)
except Exception as e:
    print(f"Failed to process example.fastq: {e}")


# 3. Process the gzipped FASTQ file
print("\n--- Processing example.fastq.gz ---")
try:
    record_counter = 0
    for record in parse_fastq("example.fastq.gz"):
        record_counter += 1
        # Access tuple elements by index
        print(f"Record {record_counter}:")
        print(f"  ID:   {record[0]}")
        print(f"  Seq:  {record[1]}")
        print(f"  Qual: {record[3]}")
        print("-" * 20)
except Exception as e:
    print(f"Failed to process example.fastq.gz: {e}")

# --- Cleanup (Optional) ---
# import os
# os.remove("example.fastq")
# os.remove("example.fastq.gz")
# print("\nCleaned up dummy files.")
