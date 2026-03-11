import polars as pl
from Bio import SeqIO
import re

def extract_primer_variants(fastq_file: str, primer_regex: str) -> pl.DataFrame:
    records = []
    compiled_regex = re.compile(primer_regex)
    
    for record in SeqIO.parse(fastq_file, "fastq"):
        sequence = str(record.seq)
        
        # Limit search to the first 50 bases to optimize performance
        match = compiled_regex.search(sequence[:50])
        
        if match:
            wobble_bases = match.group(1) 
            records.append({
                "Read_ID": record.id,
                "Primer_Variant": wobble_bases
            })
            
    return pl.DataFrame(records)

def calculate_variant_enrichment(
    df_pre_hcdr3: pl.DataFrame, 
    df_post_hcdr3: pl.DataFrame,
    pre_fastq: str,
    post_fastq: str,
    primer_regex: str
) -> pl.DataFrame:
    # 1. Extract primer variants from both FASTQ files
    df_pre_primers = extract_primer_variants(pre_fastq, primer_regex)
    df_post_primers = extract_primer_variants(post_fastq, primer_regex)
    
    # 2. Merge HCDR3 annotations with their respective primer variants
    # Assumes df_pre_hcdr3 and df_post_hcdr3 have columns: ['Read_ID', 'HCDR3']
    df_pre_full = df_pre_hcdr3.join(df_pre_primers, on="Read_ID", how="inner")
    df_post_full = df_post_hcdr3.join(df_post_primers, on="Read_ID", how="inner")
    
    # 3. Aggregate counts by HCDR3 and Primer_Variant
    df_pre_counts = df_pre_full.group_by(["HCDR3", "Primer_Variant"]).count().rename({"count": "count_pre"})
    df_post_counts = df_post_full.group_by(["HCDR3", "Primer_Variant"]).count().rename({"count": "count_post"})
    
    # 4. Full outer join to preserve sequences unique to either library
    merged_df = df_pre_counts.join(
        df_post_counts, 
        on=["HCDR3", "Primer_Variant"], 
        how="full", 
        coalesce=True
    ).with_columns([
        pl.col("count_pre").fill_null(0),
        pl.col("count_post").fill_null(0)
    ])
    
    # 5. Calculate frequencies and enrichment (Log2FC)
    total_pre = merged_df["count_pre"].sum()
    total_post = merged_df["count_post"].sum()
    pseudocount = 1e-6
    
    enrichment_df = merged_df.with_columns([
        (pl.col("count_pre") / total_pre).alias("freq_pre"),
        (pl.col("count_post") / total_post).alias("freq_post")
    ]).with_columns(
        (
            (pl.col("freq_post") + pseudocount) / (pl.col("freq_pre") + pseudocount)
        ).log(base=2).alias("log2fc")
    )
    
    # 6. Summarize statistics per HCDR3
    enrichment_summary = enrichment_df.group_by("HCDR3").agg([
        pl.len().alias("primer_variant_count"),
        pl.col("log2fc").mean().alias("mean_log2fc"),
        pl.col("log2fc").std().alias("std_dev_log2fc"),
        pl.col("count_post").sum().alias("total_post_reads")
    ]).sort("mean_log2fc", descending=True)
    
    return enrichment_summary

# --- Execution ---
# Example regex expects a prefix, exactly 3 degenerate bases, and a suffix.
# primer_pattern = r"CAGGT(.{3})CAGCT"

# summary_results = calculate_variant_enrichment(
#     df_pre_hcdr3=my_pre_annotations, 
#     df_post_hcdr3=my_post_annotations,
#     pre_fastq="sample-WPDL.fastq",
#     post_fastq="sample.fastq",
#     primer_regex=primer_pattern
# )
# print(summary_results.head(10))
