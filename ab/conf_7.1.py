#!/usr/bin/env python3
import sys
import polars as pl
from pathlib import Path

heavy_output = sys.argv[1]
light_output = sys.argv[2]

heavy_df = pl.read_csv(heavy_output, separator="\t")
light_df = pl.read_csv(light_output, separator="\t")

print(heavy_df.head(5))
print(light_df.head(5))

# FIX 3: Added () to header.strip() — was a method reference, not a call
headers_heavy = [header + "_heavy" if header.strip() else header for header in heavy_df.columns]
heavy_renamed = heavy_df.rename({old: new for old, new in zip(heavy_df.columns, headers_heavy)})

headers_light = [header + "_light" if header.strip() else header for header in light_df.columns]
light_renamed = light_df.rename({old: new for old, new in zip(light_df.columns, headers_light)})

heavy_clean = heavy_renamed.with_columns(
    pl.col("sequence_id_heavy").str.extract(r"(sequence_\d+)").alias("seq_key")
)
light_clean = light_renamed.with_columns(
    pl.col("sequence_id_light").str.extract(r"(sequence_\d+)").alias("seq_key")
)

merged_df = heavy_clean.join(light_clean, left_on="seq_key", right_on="seq_key", how="inner")
print(merged_df.head(5))

merged_subset = merged_df.select(
    pl.col("sequence_id_heavy"),
    pl.col("sequence_alignment_heavy"),
    pl.col("sequence_aa_heavy"),
    pl.col("cdr3_aa_heavy"),
    pl.col("sequence_alignment_light"),
    pl.col("sequence_aa_light"),
    pl.col("cdr3_aa_light"),
    pl.col("stop_codon_heavy"),
    pl.col("stop_codon_light")
)

merged_al_cleaned = merged_subset.with_columns(
    pl.col(["sequence_alignment_heavy", "sequence_alignment_light"])
    .str.replace_all("-", "", literal=True)
)

print(f"Number of rows before alignment gaps are removed: {len(merged_subset)}")
print(f"Number of rows after alignment gaps are removed: {len(merged_al_cleaned)}")

# FIX 2: Invalid polars filter syntax — keyword args don't work here
msub_no_stop_codons = merged_al_cleaned.filter(
    (pl.col("stop_codon_heavy") == "F") & (pl.col("stop_codon_light") == "F")
)

rows_with_stop_codons = len(merged_subset)
rows_without_stop_codons = len(msub_no_stop_codons)
print(f"Number of rows before removing stop codons: {rows_with_stop_codons}")
print(f"Number of rows after removing stop codons: {rows_without_stop_codons}")
print(f"Number of rows with stop codons (removed): {rows_with_stop_codons - rows_without_stop_codons}")

msub_sc_clean = msub_no_stop_codons.drop(["stop_codon_heavy", "stop_codon_light"])

msub_sc_2_clean = msub_sc_clean.with_columns(
    pl.col(["sequence_aa_heavy", "cdr3_aa_heavy", "sequence_aa_light", "cdr3_aa_light"])
    .str.replace_all("*", "", literal=True)
)

print(f"Number of rows before removing (*): {len(msub_sc_clean)}")
print(f"Number of rows after removing (*): {len(msub_sc_2_clean)}")

msub_null_clean = msub_sc_2_clean.drop_nulls()
rows_with_empty = len(msub_sc_2_clean)
rows_without_empty = len(msub_null_clean)
print(f"Number of rows before removing empty cells: {rows_with_empty}")
print(f"Number of rows after removing rows with empty cells: {rows_without_empty}")
print(f"Number of rows with empty values or nulls (removed): {rows_with_empty - rows_without_empty}")

# FIX 4: Nested same-type quotes inside f-string — use single quotes inside
print(f"Memory: {msub_null_clean.estimated_size('mb'):.2f} megabytes")

# FIX 1: Create the output directory BEFORE trying to write files into it
path = Path().cwd() / "igblast_tsv_files"
path.mkdir(exist_ok=True)
print(f"Path where TSV files will be stored: {path}")

top_20_hcdr3s = msub_null_clean["cdr3_aa_heavy"].value_counts(sort=True).head(20)
print(top_20_hcdr3s)
top_20_hcdr3s.write_csv("top_20_hcdr3s.csv")

top_20_seqs = top_20_hcdr3s["cdr3_aa_heavy"].to_list()

for i, seq in enumerate(top_20_seqs, 1):
    cluster_df = msub_null_clean.filter(pl.col("sequence_aa_heavy").str.contains(seq))
    filename = path / f"{seq}_cluster.tsv"  # also fixed to use Path object
    cluster_df.write_csv(filename, separator="\t")

top_20_lcdr3s = msub_null_clean["cdr3_aa_light"].value_counts(sort=True).head(20)
print(top_20_lcdr3s)
top_20_lcdr3s.write_csv("top_20_lcdr3s.csv")

suffix = "_cluster.tsv"
heavy_fasta = path / "heavy"
heavy_fasta.mkdir(exist_ok=True)
light_fasta = path / "light"
light_fasta.mkdir(exist_ok=True)

for tsv in path.glob("*.tsv"):
    lcdr3_groups = {}
    lcdr3_groups_2 = {}
    prefix = tsv.name.replace(suffix, "")
    tsv_df = pl.read_csv(tsv, separator="\t")

    for i, seq in enumerate(tsv_df.iter_rows(named=True)):
        if seq["cdr3_aa_light"] not in lcdr3_groups:
            lcdr3_groups[seq["cdr3_aa_light"]] = []
        # FIX 4: Single quotes inside f-strings
        lcdr3_groups[seq["cdr3_aa_light"]].append(f">{seq['sequence_id_heavy']}\n{seq['sequence_alignment_heavy']}\n")

    for cdr3, records in lcdr3_groups.items():
        output_filename = f"{prefix}_{cdr3}.fasta"
        output_filepath = heavy_fasta / output_filename
        with open(output_filepath, 'w') as outfile:
            outfile.writelines(records)

    for i, seq in enumerate(tsv_df.iter_rows(named=True)):
        if seq["cdr3_aa_light"] not in lcdr3_groups_2:
            lcdr3_groups_2[seq["cdr3_aa_light"]] = []
        # FIX 4: Single quotes inside f-strings
        lcdr3_groups_2[seq["cdr3_aa_light"]].append(f">{seq['sequence_id_heavy']}\n{seq['sequence_alignment_light']}\n")

    for cdr3, records in lcdr3_groups_2.items():
        output_filename = f"{prefix}_{cdr3}.fasta"
        output_filepath = light_fasta / output_filename
        with open(output_filepath, 'w') as outfile:
            outfile.writelines(records)

fasta_above_cut_off = 0
count_cut_off = 20
for fasta in heavy_fasta.glob("*.fasta"):
    line_count = 0
    with open(fasta, 'r') as fa:
        for line in fa:
            line_count += 1
    if (line_count / 2) > count_cut_off:
        fasta_above_cut_off += 1

print(f"There are {fasta_above_cut_off} files with more than {count_cut_off} sequences")
# FIX 4: Single quotes inside f-string
print(f"Number of FASTA files: {len(list(heavy_fasta.glob('*.fasta')))}")
