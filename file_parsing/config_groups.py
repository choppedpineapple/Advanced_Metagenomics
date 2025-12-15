import os
import csv
import glob
from pathlib import Path

def split_tsv_to_fasta(file_pattern="*_cdr3_seqs.tsv"):
    # Find all matching files
    tsv_files = glob.glob(file_pattern)
    
    if not tsv_files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(tsv_files)} files to process. RIP your hard drive.")

    for tsv_file in tsv_files:
        process_single_file(tsv_file)

def process_single_file(filename):
    # 1. Parse filename to get the prefix
    # We expect 'XGTAWTHXSGDKS_cdr3_seqs.tsv', we want 'XGTAWTHXSGDKS'
    path_obj = Path(filename)
    original_filename = path_obj.name
    
    # Check if the suffix exists to avoid index errors
    suffix = "_cdr3_seqs.tsv"
    if not original_filename.endswith(suffix):
        print(f"Skipping {filename}: Does not end with '{suffix}'")
        return

    prefix = original_filename.replace(suffix, "")
    
    # 2. Create the subdirectory
    output_dir = path_obj.parent / prefix
    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory {output_dir}: {e}")
        return

    print(f"Processing: {filename} -> Output dir: {output_dir}")

    # 3. Read and Group Data in Memory
    # Structure: groups = { 'CDR3_SEQ': [ ('header1', 'seq1'), ('header2', 'seq2') ] }
    groups = {}
    
    try:
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            
            # Skip the header
            # Header: seq_id_heavy, seq_aa_heavy, seq_id_light, seq_aa_light, cdr3_aa_light
            next(reader, None) 
            
            for row_idx, row in enumerate(reader):
                if len(row) < 5:
                    continue # Skip malformed lines
                
                # Extract fields
                header_id = row[0]
                seq_heavy = row[1]
                seq_light = row[3]
                cdr3_aa = row[4].strip()
                
                # Construct sequence (Strictly adjacent)
                full_sequence = seq_heavy + seq_light
                
                # Grouping
                if cdr3_aa not in groups:
                    groups[cdr3_aa] = []
                groups[cdr3_aa].append(f">{header_id}\n{full_sequence}\n")
                
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        return

    # 4. Write Groups to Files
    # This is the part where we flood the directory
    for cdr3, records in groups.items():
        # Sanitize filename just in case CDR3 has weird chars (unlikely for AA, but still)
        # We append the CDR3 sequence to the original prefix
        output_filename = f"{prefix}_{cdr3}.fasta"
        output_path = output_dir / output_filename
        
        try:
            with open(output_path, 'w') as f_out:
                f_out.writelines(records)
        except OSError:
            print(f"Warning: Could not write file for CDR3: {cdr3}. Filename might be too long?")

    print(f"Finished {filename}: Created {len(groups)} FASTA files.")

if __name__ == "__main__":
    # Run it
    split_tsv_to_fasta()
