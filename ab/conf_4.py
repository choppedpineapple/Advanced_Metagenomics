import polars as pl

# Get top 20 most occurring CDR3s with counts
top_20 = (
    merged_df
    .group_by("cdr3_aa_heavy")
    .agg(pl.count().alias("count"))
    .sort("count", descending=True)
    .head(20)
)

# Write the top 20 to CSV
top_20.write_csv("top_20_cdr3s.csv")

# Iterate through top 20 and create separate TSV files
for idx, row in enumerate(top_20.iter_rows(named=True), start=1):
    cdr3_seq = row["cdr3_aa_heavy"]
    count = row["count"]
    
    # Filter merged_df for this specific CDR3
    cluster_df = merged_df.filter(pl.col("cdr3_aa_heavy") == cdr3_seq)
    
    # Write to TSV - using index to avoid filename issues
    cluster_df.write_csv(f"cluster_{idx:02d}_n{count}.tsv", separator="\t")
