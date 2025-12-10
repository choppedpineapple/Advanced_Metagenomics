import pandas as pd
from pathlib import Path

def to_fasta(df, seq_col, id_col, out_path):
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            f.write(f">{row[id_col]}\n{row[seq_col]}\n")

def match_and_export(df1, df2):
    # ensure amino_acid is string
    df2["amino_acid"] = df2["amino_acid"].astype(str)

    # group df1 by light chain CDR3 (the grouping unit)
    for light_cdr3, group in df1.groupby("cdr3_aa_light"):

        combined_matches = []  # store all df2 matches for this light-chain group

        for _, row in group.iterrows():
            heavy_cdr3 = row["cdr3_aa"]

            # substring match
            matches = df2[df2["amino_acid"].str.contains(heavy_cdr3, na=False)]

            # store which heavy CDR3 caused this match
            matches = matches.copy()
            matches["matched_heavy_cdr3"] = heavy_cdr3

            if not matches.empty:
                combined_matches.append(matches)

        if combined_matches:
            result = pd.concat(combined_matches).drop_duplicates()

            # filename: based on light-chain cdr3
            safe_name = light_cdr3.replace("*", "_").replace("/", "_")
            out_path = Path(f"{safe_name}_matches.fasta")

            # FASTA uses amino_acid as seq and index or id from df2 as header
            to_fasta(result.reset_index(), seq_col="amino_acid", id_col="index", out_path=out_path)

            print(f"Saved: {out_path}  (n={len(result)})")
        else:
            print(f"No matches for light-chain group: {light_cdr3}")
