#!/usr/bin/env python3

import pandas as pd
from collections import Counter

# ---- CONFIG ----
input_csv = "igblast_output.csv"          # your IgBlast CSV
cdr3_column = "CDR3_nt"                   # change if your column name differs
full_seq_column = "sequence"              # change if your seq column differs
output_csv = "cdr3_ranked_clones.csv"
# ----------------

# Load IgBlast CSV
df = pd.read_csv(input_csv)

# Drop rows with missing CDR3
df = df.dropna(subset=[cdr3_column])

# Count how many times each CDR3 appears
cdr3_counts = Counter(df[cdr3_column])

# Build final table
records = []
for cdr3, cnt in cdr3_counts.items():
    subset = df[df[cdr3_column] == cdr3]

    # Number of UNIQUE full sequences for that CDR3 (optional, but labs love it)
    unique_full = subset[full_seq_column].nunique()

    records.append({
        "CDR3_nt": cdr3,
        "count": cnt,
        "unique_full_sequences": unique_full
    })

# Convert to dataframe
out_df = pd.DataFrame(records)

# Sort by abundance
out_df = out_df.sort_values("count", ascending=False)

# Save
out_df.to_csv(output_csv, index=False)

print("Done! Output saved to:", output_csv)
