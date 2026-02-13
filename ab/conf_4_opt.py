import polars as pl

top_20 = (
    merged_df["cdr3_aa_heavy"]
    .value_counts(sort=True)
    .head(20)
    .with_row_index("cluster_id", offset=1)
)

top_20.write_csv("top_20_cdr3_counts.csv")

for cluster_id, cdr3, count in top_20.iter_rows():
    cluster_df = merged_df.filter(pl.col("cdr3_aa_heavy") == cdr3)
    
    # Truncate sequence to 12 chars max for filename
    seq_preview = cdr3[:12] if len(cdr3) > 12 else cdr3
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in seq_preview)
    
