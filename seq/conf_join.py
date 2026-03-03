import polars as pl

# 1. Load the IgBLAST AIRR outputs (which you ran on the DEREPLICATED sequences)
df_heavy = pl.read_csv("heavy_airr.tsv", separator="\t")
df_light = pl.read_csv("light_airr.tsv", separator="\t")

# 2. Append suffixes to all columns EXCEPT the 'sequence_id' so we can join on it
heavy_rename = {col: f"{col}_heavy" for col in df_heavy.columns if col != "sequence_id"}
light_rename = {col: f"{col}_light" for col in df_light.columns if col != "sequence_id"}

df_heavy = df_heavy.rename(heavy_rename)
df_light = df_light.rename(light_rename)

# 3. Inner join the heavy and light chains
df_scfv = df_heavy.join(df_light, on="sequence_id", how="inner")

# 4. The Magic: Extract the raw count from the header and cast to integer
df_final = df_scfv.with_columns(
    pl.col("sequence_id")
    .str.extract(r"size=(\d+)", 1)
    .cast(pl.Int32)
    .alias("original_read_count")
).sort("original_read_count", descending=True)

print(df_final.head())
