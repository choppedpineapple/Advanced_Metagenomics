import pandas as pd
from pathlib import Path

# Read the TSV file
input_file = "your_input_file.tsv"  # Replace with your actual file path
df = pd.read_csv(input_file, sep='\t')

# Select only the required columns
columns_to_keep = [
    "sequence_id_heavy", 
    "sequence_heavy", 
    "sequence_id_light", 
    "sequence_light", 
    "stop_codon_heavy", 
    "stop_codon_light", 
    "cdr3_aa_heavy", 
    "cdr3_aa_light"
]

# Keep only specified columns that exist in the dataframe
available_columns = [col for col in columns_to_keep if col in df.columns]
df = df[available_columns]

# Remove rows where ANY of the selected columns is empty/NaN
df = df.dropna(subset=available_columns)

# Filter rows where both stop codons are "F"
df = df[(df['stop_codon_heavy'] == 'F') & (df['stop_codon_light'] == 'F')]

# Find top 20 most occurring cdr3_aa_heavy sequences
top_20_cdr3 = df['cdr3_aa_heavy'].value_counts().head(20)

print(f"Total rows after filtering: {len(df)}")
print(f"\nTop 20 most occurring CDR3 heavy chains:")
print(top_20_cdr3)

# Create output directory if it doesn't exist
output_dir = Path("cdr3_output")
output_dir.mkdir(exist_ok=True)

# Export each top CDR3 group to separate TSV files
for cdr3_sequence in top_20_cdr3.index:
    # Filter data for this specific CDR3
    cdr3_group = df[df['cdr3_aa_heavy'] == cdr3_sequence]
    
    # Create filename
    output_filename = output_dir / f"{cdr3_sequence}_cdr3.tsv"
    
    # Write to TSV
    cdr3_group.to_csv(output_filename, sep='\t', index=False)
    
    print(f"Exported {len(cdr3_group)} rows to {output_filename}")

print(f"\n✅ Done! All files saved to '{output_dir}' directory")
