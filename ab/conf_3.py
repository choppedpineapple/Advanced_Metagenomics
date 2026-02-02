#!/usr/bin/env python3
import sys
import pandas as pd
from pathlib import Path
import re

def extract_columns(heavy_sequence_ranking, igblast_merged, output_dir=None):
    # Set output directory
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read input files with error handling
        df_hsr = pd.read_csv(heavy_sequence_ranking, usecols=["cdr3_aa", "SequenceCount"])
        df_igb = pd.read_csv(igblast_merged, usecols=[
            "sequence_id_heavy", "sequence_aa_heavy", 
            "sequence_id_light", "sequence_aa_light", "cdr3_aa_light"
        ])
    except FileNotFoundError as e:
        print(f"Error: Input file not found - {e}")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Required column missing - {e}")
        sys.exit(1)
    
    # Sort and get top 20
    df_hsr_sorted = df_hsr.sort_values(by="SequenceCount", ascending=False)
    df_hsr_sorted_top20 = df_hsr_sorted.head(20).copy()
    
    # Remove any NaN values and convert to string
    df_hsr_sorted_top20 = df_hsr_sorted_top20.dropna(subset=["cdr3_aa"])
    df_hsr_sorted_top20["cdr3_aa"] = df_hsr_sorted_top20["cdr3_aa"].astype(str)
    
    # Also ensure sequence_aa_heavy is string type
    df_igb["sequence_aa_heavy"] = df_igb["sequence_aa_heavy"].astype(str)
    
    # Process each CDR3 sequence
    for i, row in enumerate(df_hsr_sorted_top20.itertuples(), start=1):
        cdr3 = row.cdr3_aa
        
        # Escape special regex characters in CDR3 sequence
        cdr3_escaped = re.escape(cdr3)
        
        # Find matches
        matches = df_igb[df_igb["sequence_aa_heavy"].str.contains(
            cdr3_escaped, na=False, regex=True
        )]
        
        if matches.empty:
            print(f"No matches found for CDR3 #{i}: {cdr3}")
            continue
        
        output_file = output_dir / f"{i}_cdr3_seqs.tsv"
        matches.to_csv(output_file, sep="\t", index=False)
        print(f"Saved {len(matches)} matches for CDR3 #{i} to {output_file}")
    
    # Process all TSV files in output directory
    tsv_files = list(output_dir.glob("*_cdr3_seqs.tsv"))
    
    if not tsv_files:
        print("Warning: No TSV files found to process")
        return
    
    for tsv_file in tsv_files:
        if not tsv_file.is_file():
            continue
        
        try:
            df = pd.read_csv(tsv_file, sep="\t")
            
            # Check if file has enough columns
            if len(df.columns) < 5:
                print(f"Warning: {tsv_file.name} has fewer than 5 columns, skipping grouping")
                continue
            
            col_name = df.columns[4]  # cdr3_aa_light column
            
            # Group by the 5th column
            grouped = df.groupby(col_name)
            
            for key, group_df in grouped:
                # Sanitize the key for filename
                safe_key = str(key).replace(" ", "_").replace("/", "_").replace("\\", "_")
                # Remove other potentially problematic characters
                safe_key = re.sub(r'[^\w\-_.]', '_', safe_key)
                
                output_file = output_dir / f"{tsv_file.stem}_{safe_key}.tsv"
                group_df.to_csv(output_file, sep="\t", index=False)
                print(f"Saved group '{key}' ({len(group_df)} rows) to {output_file.name}")
        
        except Exception as e:
            print(f"Error processing {tsv_file.name}: {e}")
            continue

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <heavy_sequence_ranking.csv> <igblast_merged.csv> [output_dir]")
        sys.exit(1)
    
    heavy_sequence_ranking = sys.argv[1]
    igblast_merged = sys.argv[2]
    output_dir = sys.argv[3] if len(sys.argv) > 3 else None
    
    extract_columns(heavy_sequence_ranking, igblast_merged, output_dir)

if __name__ == "__main__":
    main()
