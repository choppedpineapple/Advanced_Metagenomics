import sys
import os

class FastFastqParser:
    """
    A high-performance FASTQ parser that utilizes binary buffering and 
    list stride slicing to minimize Python loop overhead.
    """
    
    def __init__(self, filepath, chunk_size=1024*1024*4):
        """
        :param filepath: Path to the fastq file.
        :param chunk_size: Size in bytes to read per I/O operation (default 4MB).
        """
        self.filepath = filepath
        self.chunk_size = chunk_size

    def parse(self):
        """
        Generator that yields tuples of (header, sequence, quality).
        Keeps data in bytes for maximum speed.
        """
        
        # Use low-level file descriptor for raw binary speed
        with open(self.filepath, 'rb') as f:
            remainder = b''
            
            while True:
                # Read a large chunk of bytes
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                
                # Combine with the leftover bytes from the previous chunk
                chunk = remainder + chunk
                
                # Find the last newline to ensure we don't cut a line in half
                # We slice the chunk to process only complete lines
                last_newline_idx = chunk.rfind(b'\n')
                
                if last_newline_idx == -1:
                    # Buffer is smaller than a single line (rare, but possible)
                    remainder = chunk
                    continue
                
                # Separate complete data from the remainder
                complete_data = chunk[:last_newline_idx+1]
                remainder = chunk[last_newline_idx+1:]
                
                # FAST SPLIT: Split entirely in C
                lines = complete_data.split(b'\n')
                
                # Remove the trailing empty string caused by the last newline
                if lines[-1] == b'':
                    lines.pop()

                # THE NOVEL TRICK: List Slicing with Strides
                # Instead of "for line in lines", we slice the list.
                # Standard FASTQ format is 4 lines per record.
                
                # We must ensure the list length is a multiple of 4 to use zip cleanly.
                # If the chunk boundary cut a record in the middle (e.g. between seq and qual),
                # we need to handle that logic.
                
                num_lines = len(lines)
                remainder_count = num_lines % 4
                
                if remainder_count > 0:
                    # Move the incomplete record lines back to the remainder buffer
                    # Reconstruct bytes from the orphan lines
                    orphans = lines[-remainder_count:]
                    lines = lines[:-remainder_count]
                    
                    # Add these lines back to remainder for the next iteration
                    remainder = b'\n'.join(orphans) + b'\n' + remainder
                
                # Yield result using zip on sliced lists
                # lines[0::4] -> Headers
                # lines[1::4] -> Sequences
                # lines[3::4] -> Qualities (Skip line 2 which is '+')
                
                # This loop runs 4x fewer times than a standard parser
                for header, seq, qual in zip(lines[0::4], lines[1::4], lines[3::4]):
                    yield header, seq, qual

            # Handle any remaining data (corruption or weird file ending)
            if remainder:
                lines = remainder.split(b'\n')
                if len(lines) >= 4:
                     yield lines[0], lines[1], lines[3]

# --- Usage Example ---
if __name__ == "__main__":
    import time
    
    # Create a dummy file for testing if one doesn't exist
    filename = "test_large.fastq"
    if not os.path.exists(filename):
        print(f"Generating dummy {filename}...")
        with open(filename, 'wb') as f:
            # Write 500,000 records (~50MB)
            record = b'@SEQ_ID\nAGCTAGCTAGCTAGCTAGCT\n+\n!@#$!@#$!@#$!@#$!@#$\n'
            f.write(record * 500_000)
            
    print("Starting parse...")
    start_time = time.time()
    
    parser = FastFastqParser(filename)
    
    count = 0
    seq_len_total = 0
    
    # Parsing loop
    for header, seq, qual in parser.parse():
        # NOTE: Data is returned as bytes. decode() only if strictly necessary.
        count += 1
        seq_len_total += len(seq)
        
        # Just to show it works, print first record
        if count == 1:
            print(f"Sample: {header} -> {seq[:5]}...")

    end_time = time.time()
    
    print(f"\nProcessed {count:,} records.")
    print(f"Total Sequence Length: {seq_len_total:,} bp")
    print(f"Time Taken: {end_time - start_time:.4f} seconds")
    print(f"Rate: {count / (end_time - start_time):,.0f} records/sec")
