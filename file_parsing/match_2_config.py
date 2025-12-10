import re
import pandas as pd

# ensure seqs_aa is string
df_igb["seqs_aa"] = df_igb["seqs_aa"].astype(str)

for i, cdr3 in enumerate(sorted_df["cdr3_aa"], start=1):

    # Match full rows where seqs_aa contains the CDR3 substring
    matches = df_igb[df_igb["seqs_aa"].str.contains(cdr3, na=False, regex=False)]

    if matches.empty:
        continue

    # Filename-safe CDR3 (this is why regex is used here)
    # It replaces forbidden filename characters with '_'
    safe_cdr3 = re.sub(r"[^A-Za-z0-9]", "_", cdr3)

    filename = f"cdr3_{i:03d}_{safe_cdr3}.tsv"

    # Write matching rows (full df_igb rows)
    matches.to_csv(filename, sep="\t", index=False)

    print(f"Wrote {len(matches)} rows for {cdr3} → {filename}")
