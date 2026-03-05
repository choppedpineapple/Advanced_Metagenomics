# Polars + TSV Files: The Only Guide You'll Ever Need
> Because life is too short for Pandas and its frankly criminal performance on large files.

---

## Table of Contents
1. [Why Polars? (And Why Not Pandas?)](#why-polars)
2. [Reading & Writing TSV Files](#reading--writing-tsv-files)
3. [Selecting, Filtering & Slicing](#selecting-filtering--slicing)
4. [Column Operations & Transformations](#column-operations--transformations)
5. [String Operations](#string-operations)
6. [Grouping & Aggregations](#grouping--aggregations)
7. [Joins & Merges](#joins--merges)
8. [Handling Missing Data](#handling-missing-data)
9. [Lazy Evaluation (The Secret Weapon)](#lazy-evaluation-the-secret-weapon)
10. [Bioinformatics-Specific Use Cases](#bioinformatics-specific-use-cases)
11. [Advanced Tricks & One-Liners](#advanced-tricks--one-liners)

---

## Why Polars?

Let's not pretend Pandas is fine. It isn't. Here's what the benchmarks actually say:

- Polars is **5–50× faster** than Pandas on most operations (H2O.ai benchmark, 2023)
- Uses **Apache Arrow** columnar memory format — meaning operations are cache-friendly and SIMD-vectorised
- **True multi-threading** out of the box; Pandas is largely single-threaded
- **Lazy evaluation** lets the query optimizer reorder/prune operations before touching a single byte of data
- Memory usage is typically **2–4× lower** than an equivalent Pandas workflow

```python
import polars as pl
```

That's the only import you need. No aliases. No boilerplate. Respect.

---

## Reading & Writing TSV Files

### Basic Read

```python
df = pl.read_csv("data.tsv", separator="\t")
```

### Read with Explicit Schema (always do this for large files)

```python
df = pl.read_csv(
    "data.tsv",
    separator="\t",
    schema={
        "gene_id": pl.Utf8,
        "chromosome": pl.Utf8,
        "start": pl.Int64,
        "end": pl.Int64,
        "score": pl.Float64,
        "strand": pl.Utf8,
    }
)
```

### Read Only Specific Columns (skip loading the rest entirely)

```python
df = pl.read_csv(
    "data.tsv",
    separator="\t",
    columns=["gene_id", "start", "end", "score"]
)
```

### Read with Comment Lines (common in BED/GFF files)

```python
df = pl.read_csv(
    "data.tsv",
    separator="\t",
    comment_prefix="#"
)
```

### Read Compressed TSV Directly (no decompression step needed)

```python
df = pl.read_csv("data.tsv.gz", separator="\t")
```

### Read in Chunks (streaming large files)

```python
# For files that don't fit in RAM
reader = pl.read_csv_batched("huge_file.tsv", separator="\t", batch_size=100_000)
batches = reader.next_batches(5)
df = pl.concat(batches)
```

### Write to TSV

```python
df.write_csv("output.tsv", separator="\t")
```

### Write with No Header (some tools hate headers)

```python
df.write_csv("output.tsv", separator="\t", include_header=False)
```

### Write Compressed

```python
import gzip

with gzip.open("output.tsv.gz", "wb") as f:
    df.write_csv(f, separator="\t")
```

---

## Selecting, Filtering & Slicing

### Select Columns

```python
# By name
df.select(["gene_id", "score", "chromosome"])

# By data type (grab all numeric columns)
df.select(pl.col(pl.Float64, pl.Int64))

# Exclude columns
df.select(pl.exclude("some_useless_column"))
```

### Filter Rows

```python
# Simple filter
df.filter(pl.col("score") > 50.0)

# Multiple conditions (AND)
df.filter(
    (pl.col("score") > 50.0) &
    (pl.col("chromosome") == "chr1")
)

# OR condition
df.filter(
    (pl.col("chromosome") == "chr1") |
    (pl.col("chromosome") == "chrX")
)

# Filter with isin (membership test)
chroms_of_interest = ["chr1", "chr2", "chr3"]
df.filter(pl.col("chromosome").is_in(chroms_of_interest))

# Filter nulls out
df.filter(pl.col("score").is_not_null())
```

### Slice Rows

```python
# First N rows
df.head(20)

# Last N rows
df.tail(20)

# Arbitrary slice
df.slice(100, 500)  # offset=100, length=500

# Sample randomly
df.sample(n=1000, seed=42)
df.sample(fraction=0.1, seed=42)
```

---

## Column Operations & Transformations

### Add a New Column

```python
# Constant value
df.with_columns(pl.lit("hg38").alias("genome_build"))

# Computed from existing columns
df.with_columns(
    (pl.col("end") - pl.col("start")).alias("feature_length")
)
```

### Multiple New Columns at Once

```python
df.with_columns([
    (pl.col("end") - pl.col("start")).alias("length"),
    (pl.col("score") / pl.col("score").max()).alias("norm_score"),
    pl.col("gene_id").str.to_uppercase().alias("gene_id_upper"),
])
```

### Rename Columns

```python
df.rename({"gene_id": "ensembl_id", "score": "alignment_score"})
```

### Cast Column Types

```python
df.with_columns([
    pl.col("start").cast(pl.Int32),
    pl.col("score").cast(pl.Float32),
    pl.col("strand").cast(pl.Categorical),  # saves memory on repetitive strings
])
```

### Drop Columns

```python
df.drop(["useless_col1", "useless_col2"])
```

### Sort

```python
# Single column
df.sort("score", descending=True)

# Multiple columns
df.sort(["chromosome", "start"], descending=[False, False])
```

### Apply Math Operations

```python
df.with_columns([
    pl.col("score").log(base=10.0).alias("log10_score"),
    pl.col("score").sqrt().alias("sqrt_score"),
    pl.col("score").abs().alias("abs_score"),
    (pl.col("score") * 1000).round(2).alias("score_scaled"),
])
```

### Conditional Column (if/else logic)

```python
df.with_columns(
    pl.when(pl.col("strand") == "+")
    .then(pl.lit("forward"))
    .when(pl.col("strand") == "-")
    .then(pl.lit("reverse"))
    .otherwise(pl.lit("unknown"))
    .alias("strand_label")
)
```

### Clip Values

```python
df.with_columns(
    pl.col("score").clip(lower_bound=0.0, upper_bound=1000.0)
)
```

---

## String Operations

Polars string ops are vectorised and stupidly fast. No `.apply()` hacks needed.

### Basic String Manipulation

```python
df.with_columns([
    pl.col("gene_id").str.to_uppercase().alias("gene_upper"),
    pl.col("gene_id").str.to_lowercase().alias("gene_lower"),
    pl.col("gene_id").str.strip_chars().alias("gene_stripped"),
    pl.col("gene_id").str.len_chars().alias("id_length"),
])
```

### Split a Column

```python
# Split "ENSG00000139618.15" into ID and version
df.with_columns(
    pl.col("gene_id").str.split(".").list.first().alias("base_id"),
    pl.col("gene_id").str.split(".").list.last().alias("version"),
)
```

### Extract with Regex

```python
# Extract chromosome number from "chr1", "chr22", "chrX"
df.with_columns(
    pl.col("chromosome").str.extract(r"chr(\w+)", group_index=1).alias("chrom_num")
)
```

### Replace with Regex

```python
df.with_columns(
    pl.col("gene_id").str.replace(r"ENSG0+", "G").alias("short_id")
)
```

### Check if String Contains Pattern

```python
df.filter(pl.col("gene_id").str.contains("BRCA"))
```

### Concatenate Strings (build a genomic coordinate string)

```python
df.with_columns(
    pl.concat_str([
        pl.col("chromosome"),
        pl.lit(":"),
        pl.col("start").cast(pl.Utf8),
        pl.lit("-"),
        pl.col("end").cast(pl.Utf8),
    ]).alias("region")
)
# Output: "chr1:10000-20000"
```

---

## Grouping & Aggregations

### Basic GroupBy

```python
df.group_by("chromosome").agg(
    pl.col("score").mean().alias("mean_score"),
    pl.col("score").max().alias("max_score"),
    pl.col("gene_id").count().alias("n_genes"),
)
```

### Multiple Aggregations

```python
df.group_by(["chromosome", "strand"]).agg([
    pl.col("score").mean().alias("mean_score"),
    pl.col("score").std().alias("std_score"),
    pl.col("score").median().alias("median_score"),
    pl.col("score").quantile(0.95).alias("p95_score"),
    pl.col("start").min().alias("min_start"),
    pl.col("end").max().alias("max_end"),
    pl.len().alias("count"),
])
```

### Value Counts (frequency table)

```python
df["chromosome"].value_counts(sort=True)
```

### Rolling / Window Functions

```python
# Moving average over sorted data
df.sort("start").with_columns(
    pl.col("score")
    .rolling_mean(window_size=5)
    .alias("rolling_avg_score")
)
```

### Rank Within Groups

```python
df.with_columns(
    pl.col("score")
    .rank(method="dense", descending=True)
    .over("chromosome")
    .alias("rank_within_chrom")
)
```

### Pivot (wide format)

```python
df.pivot(
    values="score",
    index="gene_id",
    on="sample_id",
    aggregate_function="mean"
)
```

---

## Joins & Merges

Polars joins are hash-based and can use multiple threads. Pandas' merge is... also a thing, I guess.

### Inner Join

```python
df1.join(df2, on="gene_id", how="inner")
```

### Left Join

```python
df1.join(df2, on="gene_id", how="left")
```

### Join on Multiple Keys

```python
df1.join(df2, on=["chromosome", "start", "end"], how="inner")
```

### Join on Different Column Names

```python
df1.join(df2, left_on="gene_id", right_on="ensembl_id", how="left")
```

### Anti-Join (rows in df1 NOT in df2)

```python
df1.join(df2, on="gene_id", how="anti")
```

### Cross Join (all combinations — use with extreme caution)

```python
df1.join(df2, how="cross")
```

### Concatenate DataFrames Vertically

```python
pl.concat([df1, df2, df3])  # Stack rows
```

### Concatenate Horizontally

```python
pl.concat([df1, df2], how="horizontal")  # Paste columns side-by-side
```

---

## Handling Missing Data

### Detect Nulls

```python
df.null_count()          # Count nulls per column
df.filter(pl.col("score").is_null())   # Rows where score is null
```

### Drop Rows with Nulls

```python
df.drop_nulls()                        # Any null in any column
df.drop_nulls(subset=["score"])        # Only if score is null
```

### Fill Nulls

```python
df.with_columns(pl.col("score").fill_null(0.0))

# Fill with column mean
df.with_columns(
    pl.col("score").fill_null(pl.col("score").mean())
)

# Forward fill (useful for time series or sorted genomic data)
df.with_columns(pl.col("score").forward_fill())

# Backward fill
df.with_columns(pl.col("score").backward_fill())
```

### Replace NaN (not the same as null in Polars!)

```python
df.with_columns(pl.col("score").fill_nan(0.0))
```

---

## Lazy Evaluation (The Secret Weapon)

This is where Polars becomes genuinely magical. With `lazy()`, you build a *query plan* and Polars optimises it before execution — predicate pushdown, projection pushdown, the works.

### Basic Lazy Query

```python
result = (
    pl.scan_csv("huge_file.tsv", separator="\t")  # Never loads full file!
    .filter(pl.col("score") > 100)
    .select(["gene_id", "chromosome", "score"])
    .with_columns(
        (pl.col("score") / pl.col("score").max()).alias("norm_score")
    )
    .sort("norm_score", descending=True)
    .collect()  # Only NOW does it execute
)
```

### Explain the Query Plan (for debugging and optimization nerd joy)

```python
plan = (
    pl.scan_csv("huge_file.tsv", separator="\t")
    .filter(pl.col("score") > 100)
    .select(["gene_id", "score"])
)
print(plan.explain(optimized=True))
```

### Streaming Mode (process data larger than RAM)

```python
result = (
    pl.scan_csv("truly_massive_file.tsv", separator="\t")
    .filter(pl.col("chromosome").is_in(["chr1", "chr2"]))
    .group_by("chromosome")
    .agg(pl.col("score").mean())
    .collect(streaming=True)  # Won't blow up your RAM
)
```

---

## Bioinformatics-Specific Use Cases

### Parse a BED file (headerless TSV)

```python
bed_schema = {
    "chrom": pl.Utf8,
    "chromStart": pl.Int64,
    "chromEnd": pl.Int64,
    "name": pl.Utf8,
    "score": pl.Int32,
    "strand": pl.Utf8,
}

bed = pl.read_csv(
    "peaks.bed",
    separator="\t",
    has_header=False,
    new_columns=list(bed_schema.keys()),
    schema_overrides=bed_schema,
    comment_prefix="#"
)
```

### Compute Feature Lengths and GC-like Stats

```python
bed.with_columns([
    (pl.col("chromEnd") - pl.col("chromStart")).alias("feature_length"),
    ((pl.col("chromEnd") - pl.col("chromStart")) / 1000).alias("feature_kb"),
])
```

### Filter Low-Score Peaks (ChIP-seq style)

```python
high_confidence_peaks = bed.filter(pl.col("score") >= 500)
```

### Parse a VCF-like TSV (variant table)

```python
vcf_df = pl.read_csv(
    "variants.tsv",
    separator="\t",
    comment_prefix="##",
    schema_overrides={
        "CHROM": pl.Utf8,
        "POS": pl.Int64,
        "REF": pl.Utf8,
        "ALT": pl.Utf8,
        "QUAL": pl.Float64,
        "AF": pl.Float64,
    }
)

# Filter SNPs only (REF and ALT are single bases)
snps = vcf_df.filter(
    (pl.col("REF").str.len_chars() == 1) &
    (pl.col("ALT").str.len_chars() == 1)
)

# Bin variants by allele frequency
snps.with_columns(
    pl.when(pl.col("AF") < 0.01).then(pl.lit("rare"))
    .when(pl.col("AF") < 0.05).then(pl.lit("low_freq"))
    .otherwise(pl.lit("common"))
    .alias("af_category")
)
```

### Gene Expression Matrix (RNA-seq counts TSV)

```python
expr = pl.read_csv("counts.tsv", separator="\t")

# Compute CPM (Counts Per Million) for each sample
sample_cols = [c for c in expr.columns if c != "gene_id"]

total_counts = expr.select(
    [pl.col(c).sum().alias(c) for c in sample_cols]
)

# Normalize
cpm = expr.with_columns([
    ((pl.col(c) / total_counts[c][0]) * 1e6).alias(f"{c}_cpm")
    for c in sample_cols
])
```

### Filter Genes by Expression Threshold

```python
# Keep genes expressed in at least 3 samples above CPM threshold
cpm_cols = [c for c in cpm.columns if c.endswith("_cpm")]
cpm_threshold = 1.0

expressed = cpm.filter(
    pl.sum_horizontal([
        (pl.col(c) > cpm_threshold).cast(pl.Int32)
        for c in cpm_cols
    ]) >= 3
)
```

### Compute Summary Stats per Gene Across Samples

```python
cpm.with_columns([
    pl.mean_horizontal(cpm_cols).alias("mean_cpm"),
    pl.max_horizontal(cpm_cols).alias("max_cpm"),
])
```

### BLAST/Alignment TSV Result Parsing

```python
blast_cols = [
    "qseqid", "sseqid", "pident", "length",
    "mismatch", "gapopen", "qstart", "qend",
    "sstart", "send", "evalue", "bitscore"
]

blast = pl.read_csv(
    "blast_results.tsv",
    separator="\t",
    has_header=False,
    new_columns=blast_cols,
    schema_overrides={"evalue": pl.Float64, "pident": pl.Float64}
)

# Best hit per query (lowest e-value)
best_hits = (
    blast
    .sort("evalue")
    .group_by("qseqid")
    .agg(pl.first("sseqid"), pl.first("evalue"), pl.first("pident"))
)

# Filter: identity >= 90%, e-value < 1e-5
filtered = blast.filter(
    (pl.col("pident") >= 90.0) &
    (pl.col("evalue") < 1e-5)
)
```

---

## Advanced Tricks & One-Liners

### Profile a DataFrame Instantly

```python
print(df.describe())
```

### Check Schema

```python
print(df.schema)
print(df.dtypes)
```

### Unique Rows

```python
df.unique()
df.unique(subset=["gene_id", "chromosome"])
```

### Explode a List Column

```python
# If a column contains lists like ["GO:001", "GO:002"]
df.with_columns(
    pl.col("go_terms").str.split(";")
).explode("go_terms")
```

### Melt (Wide → Long format)

```python
df.unpivot(
    on=["sample1", "sample2", "sample3"],
    index="gene_id",
    variable_name="sample",
    value_name="expression"
)
```

### Apply a Custom Function (when you truly must)

```python
# Use map_elements sparingly — it breaks out of Polars' vectorised engine
df.with_columns(
    pl.col("sequence").map_elements(
        lambda s: sum(1 for c in s if c in "GC") / len(s),
        return_dtype=pl.Float64
    ).alias("gc_content")
)
```

### Write Multiple TSVs Split by Chromosome

```python
for chrom, group in df.group_by("chromosome"):
    group.write_csv(f"output_{chrom[0]}.tsv", separator="\t")
```

### Read Multiple TSV Files and Concatenate

```python
import glob

files = glob.glob("data/*.tsv")
df = pl.concat([pl.read_csv(f, separator="\t") for f in files])
```

### Convert to/from Pandas (when the ecosystem forces your hand)

```python
# Polars → Pandas
df_pandas = df.to_pandas()

# Pandas → Polars
df_polars = pl.from_pandas(df_pandas)
```

### Export to Parquet (for when TSV just isn't good enough)

```python
df.write_parquet("data.parquet", compression="zstd")
df_back = pl.read_parquet("data.parquet")
```

> **Hot take:** If you're storing large tabular bioinformatics data as TSV in 2024, you should genuinely reconsider your life choices. Parquet is 5–20× smaller and 10–100× faster to query. TSV is fine for interchange; it's a terrible storage format.

---

## Quick Reference Cheat Sheet

| Task | Polars |
|---|---|
| Read TSV | `pl.read_csv("f.tsv", separator="\t")` |
| Write TSV | `df.write_csv("f.tsv", separator="\t")` |
| Filter rows | `df.filter(pl.col("x") > 5)` |
| Add column | `df.with_columns(expr.alias("name"))` |
| Group + agg | `df.group_by("col").agg(pl.col("x").mean())` |
| Join | `df1.join(df2, on="key", how="left")` |
| Sort | `df.sort("col", descending=True)` |
| Drop nulls | `df.drop_nulls(subset=["col"])` |
| Lazy scan | `pl.scan_csv("f.tsv", separator="\t")` |
| Collect lazy | `.collect()` or `.collect(streaming=True)` |

---

*Built with Polars. Pandas not invited.*
