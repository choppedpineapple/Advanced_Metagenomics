Great goal, and starting from “very basic” Python is totally fine—we’ll build up slowly, always with a concrete bioinformatics pipeline in mind.

Below is a road-map + mini-lessons that you can follow at your own pace. Each lesson introduces one new math concept and one new Python feature and one concrete bioinformatics task that benefits from that concept.

---

0. Mindset & Setup (today)

---

- Install Python 3.11+ and the VS Code editor (or JupyterLab if you prefer notebooks).  
- Create a folder `~/biomathpy/` and inside it two sub-folders:

  `lessons/` (code we will write) and `data/` (FASTA/FASTQ/BED files).  
- We’ll use only the standard library for the first 4–5 lessons, then add `numpy`, `pandas`, and `biopython`.

---

1. Lesson 1 – Counting & Basic Stats: GC-content of a FASTA

---

Bio use-case

In a DNA sequence, GC-content = (G+C)/(A+T+G+C). It influences Tm, PCR primers, etc. In Bash you’d typically use `grep | wc -l` tricks; in Python we’ll do it once and portably.

Math concept  
- Frequency = count / total  
- Percentage = frequency × 100  

Python concepts  
- `open()`, `for line in file:`, string methods `.upper()`, `.count()`

Starter code (save as `lessons/01_gc_content.py`):

```python
from pathlib import Path

def gc_content(path: Path) -> float:
    g = c = a = t = 0
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):          # skip FASTA header
                continue
            seq = line.strip().upper()
            g += seq.count("G")
            c += seq.count("C")
            a += seq.count("A")
            t += seq.count("T")
    total = g + c + a + t
    return (g + c) / total * 100

if __name__ == "__main__":
    fasta = Path("data/sequence.fasta")
    print(f"GC content = {gc_content(fasta):.2f}%")
```

Run it:

```bash
python lessons/01_gc_content.py
```

Exercise

Modify the code to work on multi-FASTA files and print per-sequence GC.

---

2. Lesson 2 – Lists & Loops: k-mer counting

---

Bio use-case

Count 6-mers (or any k) to build a frequency table, useful for motif discovery, assembly error profiling, etc.

Math concept  
- k-mer space size = 4ᵏ  
- Probability under uniform model = 1/4ᵏ  

Python concepts  
- `collections.Counter`, list comprehensions

Starter snippet:

```python
from collections import Counter
from pathlib import Path

def kmers(seq: str, k: int) -> list[str]:
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

def kmer_counts(path: Path, k: int) -> Counter:
    counts = Counter()
    with path.open() as fh:
        for line in fh:
            if line.startswith(">"):
                continue
            seq = line.strip().upper()
            counts.update(kmers(seq, k))
    return counts

if __name__ == "__main__":
    counts = kmer_counts(Path("data/sequence.fasta"), k=6)
    print(counts.most_common(10))
```

---

3. Lesson 3 – Floating-point & Log-scores: Phred quality scores

---

Bio use-case

FASTQ quality lines → Phred scores → probability of wrong base call. In Bash you’d `awk` the 4th line; in Python we’ll vectorise later.

Math concept  
- Q = -10 log₁₀(P)  
- P = 10^(-Q/10)  

Python concepts  
- `ord()` to convert ASCII, list comprehensions, simple functions

```python
def qual_to_prob(qchar: str) -> float:
    q = ord(qchar) - 33          # Sanger offset
    return 10 ** (-q / 10)

def avg_error_rate(fastq: Path) -> float:
    total_bases = 0
    total_prob = 0.0
    with fastq.open() as fh:
        for i, line in enumerate(fh):
            if i % 4 == 3:               # quality line
                probs = [qual_to_prob(c) for c in line.strip()]
                total_prob += sum(probs)
                total_bases += len(probs)
    return total_prob / total_bases

if __name__ == "__main__":
    print("Average error probability:", avg_error_rate(Path("data/reads.fastq")))
```

---

4. Lesson 4 – Intro to NumPy: Coverage & Sliding Windows

---

Bio use-case

Compute per-base coverage from a BED file (genomic intervals). In Bash you’d `bedtools genomecov`; in Python we’ll do it manually to learn arrays.

Math concept  
- Coverage = number of reads overlapping a base  
- Running mean in windows (smoothing)

Python concepts  
- NumPy arrays, vectorised addition, slicing

```python
import numpy as np
from pathlib import Path

def coverage_array(bed: Path, chrom_size: int) -> np.ndarray:
    cov = np.zeros(chrom_size, dtype=int)
    with bed.open() as fh:
        for line in fh:
            chrom, start, end = line.split()[:3]
            s, e = int(start), int(end)
            cov[s:e] += 1
    return cov

def sliding_mean(arr: np.ndarray, win: int) -> np.ndarray:
    kernel = np.ones(win) / win
    return np.convolve(arr, kernel, mode='same')

if __name__ == "__main__":
    cov = coverage_array(Path("data/alignments.bed"), chrom_size=1_000_000)
    smoothed = sliding_mean(cov, win=1000)
    np.savetxt("data/smooth_cov.tsv", smoothed, fmt="%.3f")
```

---

5. Lesson 5 – Statistics & Hypothesis Testing: Differential Gene Expression

---

Bio use-case

You have two CSVs of counts (Ctrl vs Treat). We’ll compute log₂ fold change and a simple T-test (later DESeq2 via Python wrapper).

Math concept  
- logFC = log2(mean_treat / mean_ctrl)  
- Welch’s t-test (unequal variance)

Python concepts  
- `pandas` DataFrame, `scipy.stats.ttest_ind`

```python
import pandas as pd
from scipy.stats import ttest_ind

ctrl = pd.read_csv("data/ctrl_counts.csv", index_col=0)
treat = pd.read_csv("data/treat_counts.csv", index_col=0)

def diff_expr(ctrl_df, treat_df):
    res = pd.DataFrame(index=ctrl_df.index)
    res["mean_ctrl"] = ctrl_df.mean(axis=1)
    res["mean_treat"] = treat_df.mean(axis=1)
    res["log2FC"] = np.log2((res["mean_treat"] + 1) / (res["mean_ctrl"] + 1))
    res["pval"] = ttest_ind(ctrl_df, treat_df, axis=1)[1]
    return res

result = diff_expr(ctrl, treat)
result.sort_values("pval").to_csv("data/de_results.csv")
```

---

6. Lesson 6 – Linear Algebra: k-mer frequency vectors & PCA

---

Bio use-case

Convert each sequence into a 4ᵏ-dimensional vector → PCA to see clustering of species or plasmids.

Math concept  
- Vector space, dot product, covariance matrix, eigenvectors  
- Dimensionality reduction

Python concepts  
- `scikit-learn` PCA

```python
from sklearn.decomposition import PCA
import numpy as np

# Assume we have a list of k-mer count dicts
def dict_to_vector(counter, k):
    alphabet = "ACGT"
    idx = { "".join(p): i for i, p in enumerate(itertools.product(alphabet, repeat=k)) }
    vec = np.zeros(len(idx))
    for kmer, cnt in counter.items():
        vec[idx[kmer]] = cnt
    return vec

vectors = np.array([dict_to_vector(kmer_counts(f, k=4), k=4) for f in fasta_files])
pca = PCA(n_components=2)
coords = pca.fit_transform(vectors)
np.savetxt("data/pca_coords.tsv", coords, delimiter="\t")
```

---

7. Lesson 7 – Object-Oriented Mini-toolkit

---

Create a small reusable package `biomathpy` with classes:

- `FastaReader`  
- `FastqReader`  
- `KmerCounter`  
- `CoverageTrack`

This replaces ad-hoc Bash one-liners with testable, documented Python modules that can be imported into Snakemake or Nextflow pipelines.

---

8. Integration into Pipelines

---

Where Python wins over pure Bash:

- Cross-platform: same code on macOS, Linux, WSL.  
- Parallelism: `concurrent.futures.ProcessPoolExecutor` or `joblib.Parallel`.  
- Logging & checkpointing: built-in `logging`, `tqdm`, `rich`.  
- Plotting: `matplotlib`, `seaborn`, `plotnine` for QC plots.  
- Reproducibility: lock dependencies via `requirements.txt` or `pyproject.toml`.

Example Snakemake rule using our earlier GC-content function:

```python
rule gc_report:
    input: "data/{sample}.fasta"
    output: "results/{sample}_gc.txt"
    run:
        from biomathpy.gc import gc_content
        gc = gc_content(Path(input[0]))
        Path(output[0]).write_text(f"{gc}\n")
```

---

Next Steps Checklist

[ ] Run lessons 1–3 on your own data.

[ ] Refactor lesson 1 into a function that can be imported.

[ ] Add `argparse` so each script becomes a CLI tool.

[ ] Write a pytest for `gc_content` (assert known GC).

[ ] Move to `numpy`/`pandas` in lessons 4–6.

[ ] Package everything into `pip install biomathpy`.

Whenever you get stuck, paste the exact error & 5 lines above it—happy to debug together.
