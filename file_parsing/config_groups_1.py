import os
import csv
import glob
from pathlib import Path
import re

def split_tsv_to_fasta_flat(file_pattern="*_cdr3_seqs.tsv"):
    # Find all matching files in the current directory (or provided path)
    tsv_files = glob.glob(file_pattern)
    
    if not tsv_files:
        print(f"No files found matching pattern: {file_pattern}")
        return

    print(f"Found {len(tsv_files)} TSV files. Preparing to clutter your folder...")

    for tsv_file in tsv_files:
        process_single_file_flat(tsv_file)

def process_single_file_flat(filename):
    path_obj = Path(filename)
    original_filename = path_obj.name
    
    # Define the directory where the TSV sits
    # This ensures outputs go to the same folder as inputs
    base_dir = path_obj.parent

    # 1. Parse filename to get the prefix
    suffix = "_cdr3_seqs.tsv"
    if not original_filename.endswith(suffix):
        print(f"Skipping {filename}: Does not end with '{suffix}'")
        return

    prefix = original_filename.replace(suffix, "")
    
    print(f"Processing: {filename}...")

    # 2. Read and Group Data in Memory
    groups = {}
    
    try:
        with open(filename, 'r', newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            
            # Skip the header
            next(reader, None) 
            
            for row in reader:
                if len(row) < 5:
                    continue 
                
                # Columns: 0=id_heavy, 1=aa_heavy, 2=id_light, 3=aa_light, 4=cdr3
                header_id = row[0]
                seq_heavy = row[1]
                seq_light = row[3]
                cdr3_aa = row[4].strip()

                if not cdr3_aa:
                    continue
                
                # Strictly adjacent concatenation
                full_sequence = seq_heavy + seq_light
                
                if cdr3_aa not in groups:
                    groups[cdr3_aa] = []
                groups[cdr3_aa].append(f">{header_id}\n{full_sequence}\n")
                
    except Exception as e:
        print(f"Failed to read {filename}: {e}")
        return

    # 3. Write Groups to Files (Same Directory)
    for cdr3, records in groups.items():
        # Sanitize CDR3 just in case it has a stop codon (*) which is illegal in filenames on Windows
        safe_cdr3 = re.sub(r'[\\/*?:"<>|]', '_', cdr3)
        
        output_filename = f"{prefix}_{safe_cdr3}.fasta"
        output_path = base_dir / output_filename
        
        try:
            with open(output_path, 'w') as f_out:
                f_out.writelines(records)
        except OSError as e:
            print(f"Error writing {output_filename}: {e}")

    print(f"Finished {filename}: Generated {len(groups)} FASTA files.")

if __name__ == "__main__":
    split_tsv_to_fasta_flat()
  
