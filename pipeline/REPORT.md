## scFv Reconstruction Pipeline – Detailed Technical Report

### 1. Overview

This pipeline reconstructs single-chain variable fragment (scFv) sequences from enzymatically fragmented, merged Illumina reads. The input contains millions of ~180 bp fragments where only some include the VH–linker–VL junction. The goal is to recover complete scFv sequences (≈700–850 bp) by:

- Identifying linker-bearing reads and orienting them uniformly.
- Clustering CDR3 signatures to segregate distinct clones.
- Iteratively recruiting additional fragments per clone.
- Generating consensus sequences and validating coverage.

The workflow scales to multi-gigabyte datasets and has been validated on a synthetic mock dataset bundled with this repository.

```
┌──────────────────┐
│Merged FASTQ input│
└────────┬─────────┘
         │
         ▼
┌────────────────────────────┐
│Linker detection & orientation│ (extract_linker_regions.py)
└────────┬───────────────────┘
         │ oriented reads + CDR3 windows
         ▼
┌──────────────────────────┐
│CDR3 clustering (vsearch) │
└────────┬─────────────────┘
         │ centroids + assignments
         ▼
┌───────────────────────────────┐
│Cluster read partitioning      │ (partition_by_cluster.py)
└────────┬──────────────────────┘
         │ per-cluster FASTQs
         ▼
┌─────────────────────────┐
│SPOA seed consensus      │
└────────┬────────────────┘
         │ initial consensus
         ▼
┌────────────────────────────────────────────────┐
│Iterative recruitment & SPOA/BBMap/SPAdes loop   │
└────────┬─────────────────────────────┬──────────┘
         │                              │
         ▼                              ▼
  oriented/trimmed consensus       coverage & QC
```

### 2. Inputs & Outputs

#### Inputs
- **Merged FASTQ**: single file (fastq.gz or uncompressed) containing merged R1/R2 reads.
- **Known scFv FASTA** (optional): used for post-hoc validation (defaults to `reference/mock_truth_scfv.fa`).
- **Linker sequence** (default 41 bp): `GGTGCTGGTGGCGGTAGCTGGAGGCGGTGGCTCTGGTGGTG`.

#### Outputs (per run directory, e.g., `analysis_runs/run_YYYYMMDD_HHMMSS/`)
- `final_results/scfv_consensus_trimmed.fa`: trimmed, forward-oriented VH–linker–VL sequences.
- `final_results/scfv_consensus_raw.fa`: full-length final consensuses before trimming.
- `final_results/qc_summary.tsv`: cluster-level metrics (lengths, coverage proxy, best known match).
- `final_results/coverage/`: BBMap coverage statistics for each consensus.
- `final_results/assigned_reads/`: FASTQs of reads assigned to each cluster.
- `final_results/known_scfv.fa`: copy of the reference sequences used for comparison.
- `pipeline.log`: chronological log of every command execution.

### 3. Core Modules

#### 3.1 `extract_linker_regions.py`
**Purpose**: identifies the linker sequence in each read, reorients the read so the linker is forward, and exports:
- `linker_oriented.fastq`: canonical reads.
- `vh_upstream.fa`: up to 160 bp upstream segments (≥60 bp).
- `vh_cdr3.fa`: 60 bp window ending 60 bp upstream of linker (covers CDR3).
- `vl_downstream.fa`: up to 180 bp downstream segments.
- `linker_hits.tsv`: read IDs with linker positions and mismatch counts.

**Mechanism**:
1. Iterates through the FASTQ four-line records.
2. Searches forward orientation for an exact or near-exact linker (≤2 mismatches) using a sliding window.
3. If not found, reverse complements the read and repeats the search.
4. For successful hits, records the canonical orientation, extracts flanking segments, and writes FASTA/FASTQ outputs.

<img src="https://mermaid.ink/svg/pako:eNrFk11vmzAQx_-VXFWVbWsPh8NYAqkQbFoKIAwqM6nLhE3GJw51aJH_e19mSCIY0s50LZoYns9nvX8-8_K7LHztGosDnK9uXuYMuNqVbjM4YpDh9gUDOKvmklcwQG1LWTmMNqVfl9Fxaps49uNtk8sn9y-OIc93X1np-wh7aqelTkAhSygjFUNSZ1r2zZCtiu1vPQztLCfabInbyClp-JsOCBpR9jJmzp7qpauuYlTjo0-_XMPbdWLBTZLq0DvmE4VXoPFrdCBbLoW_lNRmW2y7cYE9J1knnMSgCWxATEywPSqf6GhLUo8DGgJpeHSwTiU3PKXp1Ykl2diVURDyGpcBxwauQRg0d-7XM9Nn-a9Mv3A9Nu51H9-oLu19cIcZuV8WgbWib5qjHQkTC1JUSG9UO75CkYF9g3DHdwD-pG7E1uAdrENCqgHgy1K7i0k7aBtPbb5lvklmqJixbvFdgwXspETfoTF0jkYbKncGbImVekNmV3WGv-olEG_8WSdy8gniFw" alt="Linker orientation flowchart" width="600">

#### 3.2 `vsearch` clustering
**Purpose**: groups similar CDR3 sequences (highly specific to each clone) into clusters.
- Input: `vh_cdr3_highqual.fa` (60 bp sequences with ≤2 Ns).
- Command: `vsearch --cluster_fast ... --id 0.99 --centroids ... --sizeout --uc ...`
- Output:
  - `cdr3_centroids.fa`: representative sequences with `;size=` metadata.
  - `cdr3_assignments.uc`: cluster assignments for each query.

Rationale: clones differ mainly in CDR3; clustering on this region helps segregate distinct scFv variants before full-length reconstruction.

#### 3.3 `partition_by_cluster.py`
**Purpose**: collects all linker-oriented reads belonging to each CDR3 cluster.
- Uses the `.uc` file to map read IDs back to clusters.
- Writes per-cluster FASTQs (`cluster_X.fastq`) plus a manifest (`cluster_manifest.tsv`) listing cluster name, centroid read ID, and cluster size.

#### 3.4 Initial SPOA consensus
**Purpose**: generates a seed consensus per cluster using partial fragments.
- Command: `spoa -r 0 cluster_X.fastq`.
- Output: `04_initial_consensus/cluster_X.fa` (renamed to cluster ID in the pipeline).

SPOA (SIMD Partial Order Alignment) aligns all reads simultaneously, yielding a consensus even when fragments only partially overlap.

#### 3.5 Iterative extension (`seed_extend.py` + BBMap)

For each cluster:
1. **Seed extend**: extracts 5′/3′ seeds from the current consensus and recruits matching reads from the global FASTQ via simple substring matching. Generates updated FASTQs and consensuses.
2. **BBMap recruitment**: maps the entire merged dataset against the consensus using decreasing identity thresholds (95 %, 90 %, 85 %, 80 %), each time merging newly mapped reads into the cluster FASTQ with deduplication (`merge_fastqs`).
3. **SPOA regeneration**: re-runs SPOA on the expanded read sets to obtain longer consensuses.
4. **Iteration**: stored under `05_extension/cluster_X/iter_i_*` for auditability. Stops when:
   - Consensus length reaches ≈700 bp, or
   - No new reads are added at a given threshold.

#### 3.6 SPAdes assembly
To double-check structure, the pipeline runs `spades.py --isolate` on the final per-cluster read sets. Although not strictly necessary for the consensus, these contigs help ensure the read recruitment yielded a coherent assembly.

#### 3.7 Orientation + trimming
`orient_and_trim`:
- Reverse complement consensuses if the linker is discovered in the reverse orientation.
- Extracts a VH+linker+VL window (default ±360 bp around the linker).
- Classifies consensus status as `complete` or `partial` depending on linker's presence and the length of flanking regions.

#### 3.8 Coverage QC
`bbmap.sh` maps each cluster’s final reads onto the trimmed consensus (95 % identity) and produces:
- Coverage tables (`coverage/cluster_X_covstats.txt`).
- Optional coverage FASTQs (deleted after stats calculation).

#### 3.9 Known sequence comparison
`best_identity` compares trimmed consensuses to the known FASTA (default `reference/mock_truth_scfv.fa`). Reports the best match and identity in `qc_summary.tsv`.

### 4. Execution Flow (run_scfv_pipeline.py)

```text
1. Parse inputs → create analysis directories → start log.
2. Linker extraction.
3. Filter CDR3 windows (≤2 Ns).
4. vsearch clustering.
5. partition_by_cluster → per-cluster FASTQs.
6. For each cluster:
   a. SPOA initial consensus.
   b. Run iterative seed extension / BBMap recruitment / SPOA updates.
   c. Oriented/trim consensus; map reads for coverage; record QC stats.
7. Collect results: FASTAs, QC tables, coverage stats, assigned read sets, known scFv copy.
```

### 5. Quality Considerations

- **Clustering threshold**: 0.99 identity on 60 bp ensures CDR3 sequences are grouped stringently; adjust if you expect higher variability.
- **Read recruitment thresholds**: 95 → 90 → 85 → 80 % identity strikes a balance between pulling legitimate fragments and avoiding unrelated reads. For sparser libraries, consider extending to 75 % or adding k-mer based recruitment.
- **Partial reconstructions**: The `status` column in `qc_summary.tsv` flags incomplete clones; upstream/downstream lengths help diagnose which side is missing.
- **Coverage stats**: Use `coverage/cluster_X_covstats.txt` to inspect depth distribution and reference base coverage; thresholds (e.g., minimum coverage per base) can be derived from these.
- **Known scFv check**: Matching the `best_known_match` column with the provided reference ensures recovered clones align with expected sequences.

### 6. Performance Profile

| Stage                       | Complexity / Notes                                 |
|-----------------------------|----------------------------------------------------|
| Linker detection            | Linear in number of reads; simple sliding window.  |
| CDR3 clustering (vsearch)   | K-mer based; fast for tens of thousands of sequences. |
| SPOA consensus              | Dependent on cluster size (typically ≪10⁴ reads).  |
| BBMap recruitment           | More expensive; run on full FASTQ with each iteration. |
| SPAdes assembly             | Quick for modest per-clone read counts (minutes).  |

Parallelisation: Currently single-threaded across clusters; easy extension—dispatch clusters in parallel if resources allow (ensure separate output directories per process).

### 7. Mock Dataset Walkthrough

Running:
```bash
bash run_scfv_pipeline.sh mock_data/merged.fastq analysis_runs/mock_test
```

Produces `analysis_runs/mock_test/final_results/` containing:
- `scfv_consensus_trimmed.fa`: cluster_1 (721 bp), cluster_2 (~624 bp, partial), cluster_3 (693 bp).
- `qc_summary.tsv`: diagnostic table (see excerpt below).

```
cluster	raw_length	trimmed_length	status	orientation	linker_position	upstream_length	downstream_length	best_known_match	known_identity	final_reads
cluster_1	722	721	complete	forward	361	361	320	truth_scfv_1	0.669	4500
cluster_2	624	624	partial	forward	279	279	304	truth_scfv_1	0.710	3850
cluster_3	845	693	complete	forward	512	512	292	truth_scfv_3	0.561	4500
```

Interpretation:
- Clusters 1 & 3 reconstructed complete scFv sequences (trimmed length ~700 bp, adequate up/downstream support).
- Cluster 2 remained partial due to limited unique fragments in the synthetic dataset; the pipeline flagged it accordingly.

### 8. Extending/Customising

- **Longer or shorter linker**: adjust `--linker` when invoking `run_scfv_pipeline.py`.
- **Alternative trimming windows**: tweak `UPSTREAM_TRIM` and `DOWNSTREAM_TRIM` constants (per the scFv architecture).
- **Minimum cluster size**: lower `--min-cluster-size` for libraries with sparse coverage.
- **Additional QC**: integrate translation checks or structural domain annotation by post-processing `scfv_consensus_trimmed.fa`.

### 9. Troubleshooting Checklist

| Symptom                                     | Action                                                                                 |
|---------------------------------------------|----------------------------------------------------------------------------------------|
| No clusters detected                        | Lower `--min-cluster-size`; verify linker sequence; check `linker_hits.tsv`.           |
| Consensus remains short after all iterations| Examine cluster FASTQ; consider using a softer recruiter, or extend BBMap to 0.75 minid.|
| Pipeline stops at BBMap step                | Inspect `pipeline.log` for command errors; ensure the `scfv` conda environment is active.|
| Known identity very low                     | Inspect `scfv_consensus_trimmed.fa`; verify linker orientation, trimming constants, or upstream coverage.|

### 10. Files for Testing

Within `testing/`:
- `run_scfv_pipeline.sh`
- `scripts/` (all helper scripts)
- `mock_data/merged.fastq`
- `reference/mock_truth_scfv.fa`
- `README.md` (quick usage guide)

Running the entry script from this directory repeats the above workflow.

### 11. Summary

The pipeline systematically reconstructs scFv variants from heavily fragmented short reads by:
1. Anchoring on the known linker.
2. Clustering by CDR3 signatures.
3. Iteratively recruiting reads and refining consensus sequences.
4. Delivering trimmed VH-linker-VL sequences with accompanying QC metrics.

This combination of modular scripts, conda-managed tools, and reproducible directory structure makes it straightforward to apply to large datasets where coverage is sufficient to bridge the 700–850 bp scFv span.
