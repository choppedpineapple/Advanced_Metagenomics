import polars as pl

# 1. Find the top 20 most occurring cdr3s
# value_counts(sort=True) groups, counts, and sorts in one step.
top_20_df = merged_df["cdr3_aa_heavy"].value_counts(sort=True).head(20)

# 2. Write just the top 20 cdr3s (with counts) to a CSV
top_20_df.write_csv("top_20_cdr3_counts.csv")

# 3. Iterate and write rows to separate TSV files
# We convert the cdr3 column to a list to iterate over it in Python
top_sequences = top_20_df["cdr3_aa_heavy"].to_list()

for i, seq in enumerate(top_sequences, 1):
    # Filter the original dataframe for the specific cdr3
    # Note: We use pl.col to access the column for filtering
    cluster_df = merged_df.filter(pl.col("cdr3_aa_heavy") == seq)
    
    # Define filename (e.g., cluster_1.tsv, cluster_2.tsv)
    filename = f"cluster_{i}.tsv"
    
    # Write to TSV using separator='\t'
    cluster_df.write_csv(filename, separator='\t')
