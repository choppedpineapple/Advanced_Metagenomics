#!/usr/bin/env python3

import pandas as pd

# ---- CONFIG ----
input_csv = "igblast_output.csv"          # your IgBlast CSV
cdr3_column = "CDR3_nt"                   # change if needed
full_seq_column = "sequence"              # column containing full V(D)J nucleotide sequence
output_csv = "cdr3_ranked_clones.csv"
# ----------------

# Load IgBlast CSV
df = pd.read_csv(input_csv)

# Keep only rows with valid CDR3 and full sequence
df = df.dropna(subset=[cdr3_column, full_seq_column])

# Group by exact CDR3 nucleotide match
groups = df.groupby(cdr3_column)

records = []
for cdr3, grp in groups:
    total_reads = len(grp)
    unique_full = grp[full_seq_column].nunique()

    records.append({
        "CDR3_nt": cdr3,
        "count": total_reads,
        "unique_full_sequences": unique_full
    })

# Convert to dataframe
out_df = pd.DataFrame(records)

# Sort by clone abundance
out_df = out_df.sort_values("count", ascending=False)

# Save output
out_df.to_csv(output_csv, index=False)

print("Done. Ranked clones saved to:", output_csv)
