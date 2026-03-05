# 🍎 Pomegranate Python Library — A Comprehensive Guide

> **pomegranate** is a fast, flexible probabilistic modeling library built on top of PyTorch.  
> It supports GPU acceleration, missing value handling, and seamless integration with the PyTorch ecosystem.  
> Originally built by Jacob Schreiber, v1.0+ was a near-complete rewrite — so if you're following old tutorials, prepare for pain.

---

## Table of Contents

1. [Installation & Setup](#1-installation--setup)
2. [Probability Distributions](#2-probability-distributions)
3. [Gaussian Mixture Models (GMM)](#3-gaussian-mixture-models-gmm)
4. [Hidden Markov Models (HMM)](#4-hidden-markov-models-hmm)
5. [Bayesian Networks](#5-bayesian-networks)
6. [Naive Bayes Classifier](#6-naive-bayes-classifier)
7. [Markov Chains](#7-markov-chains)
8. [Factor Graphs](#8-factor-graphs)
9. [Handling Missing Data](#9-handling-missing-data)
10. [GPU Acceleration](#10-gpu-acceleration)
11. [Bioinformatics Use Cases](#11-bioinformatics-use-cases)
12. [Performance Notes](#12-performance-notes)

---

## 1. Installation & Setup

```bash
pip install pomegranate
```

For GPU support, make sure you have a CUDA-compatible PyTorch installation first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install pomegranate
```

### Basic Imports

```python
import torch
import numpy as np
import pomegranate

# Core modules you'll use repeatedly
from pomegranate.distributions import (
    Normal,
    Exponential,
    LogNormal,
    Categorical,
    Poisson,
    Uniform,
    Beta,
    Gamma,
    DirichletDistribution,
    MultivariateGaussian,
    IndependentComponents
)
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM, SparseHMM
from pomegranate.bayes_classifier import BayesClassifier
from pomegranate.naive_bayes import NaiveBayes
from pomegranate.markov_chain import MarkovChain
from pomegranate.bayesian_network import BayesianNetwork
```

> ⚠️ **Heads up:** pomegranate v1.0+ expects **PyTorch tensors**, not numpy arrays. You'll need to convert explicitly with `torch.tensor(data, dtype=torch.float32)`.

---

## 2. Probability Distributions

The building blocks of everything in pomegranate. Every model is ultimately composed of distributions.

### 2.1 Univariate Normal Distribution

```python
import torch
from pomegranate.distributions import Normal

# --- Fitting from data ---
data = torch.tensor([2.3, 3.1, 2.8, 3.5, 2.9, 3.2, 2.7], dtype=torch.float32).unsqueeze(1)

dist = Normal()
dist.fit(data)

print(f"Mean: {dist.means}")        # learned mean
print(f"Std:  {dist.covs}")         # learned std dev

# --- Manual initialization ---
dist_manual = Normal([3.0], [0.5])  # mean=3, std=0.5

# --- Log probability ---
x = torch.tensor([[2.5], [3.0], [3.5]], dtype=torch.float32)
log_probs = dist_manual.log_probability(x)
print(log_probs)

# --- Sampling ---
samples = dist_manual.sample(100)
print(samples.shape)  # (100, 1)
```

### 2.2 Multivariate Gaussian

```python
from pomegranate.distributions import MultivariateGaussian

# Gene expression across 3 genes — fitting a multivariate distribution
data = torch.randn(500, 3, dtype=torch.float32)  # 500 samples, 3 features

mvg = MultivariateGaussian()
mvg.fit(data)

print("Mean vector:", mvg.means)
print("Covariance matrix:", mvg.covs)

# Evaluate log-likelihood of new observations
new_obs = torch.tensor([[0.1, -0.2, 0.5]], dtype=torch.float32)
print("Log-prob:", mvg.log_probability(new_obs))
```

### 2.3 Categorical Distribution

Useful for discrete data — DNA bases, amino acids, tokens, etc.

```python
from pomegranate.distributions import Categorical

# DNA base frequencies: A, C, G, T
data = torch.tensor([[0], [1], [2], [3], [0], [0], [1], [3]], dtype=torch.int32)

cat = Categorical()
cat.fit(data)

print("Probabilities:", cat.probs)  # [P(A), P(C), P(G), P(T)]

# Manually set probabilities (like a known background model)
cat_manual = Categorical([[0.25, 0.25, 0.25, 0.25]])  # uniform background

log_p = cat_manual.log_probability(torch.tensor([[0], [1], [2], [3]], dtype=torch.int32))
print(log_p)  # should all be log(0.25) ≈ -1.386
```

### 2.4 Poisson Distribution

Great for count data — reads per gene, mutations per site, etc.

```python
from pomegranate.distributions import Poisson

# Read counts per gene
counts = torch.tensor([[5], [3], [8], [2], [6], [4], [7]], dtype=torch.float32)

pois = Poisson()
pois.fit(counts)

print("Lambda:", pois.lambdas)  # estimated rate

# Probability of observing exactly 5 reads
x = torch.tensor([[5.0]], dtype=torch.float32)
print("Log P(X=5):", pois.log_probability(x))
```

### 2.5 Mixed Independent Components

When features are independent but from *different* distribution families:

```python
from pomegranate.distributions import IndependentComponents, Normal, Poisson

# Feature 1: gene expression (continuous) → Normal
# Feature 2: variant count (discrete) → Poisson
d = IndependentComponents([
    Normal([0.0], [1.0]),
    Poisson([3.0])
])

# Mixed data: col 0 is float expression, col 1 is integer count
data = torch.tensor([
    [1.2, 3.0],
    [0.5, 5.0],
    [-0.3, 2.0]
], dtype=torch.float32)

log_p = d.log_probability(data)
print(log_p)
```

---

## 3. Gaussian Mixture Models (GMM)

GMMs model data as a *superposition* of k Gaussian components. The workhorse of unsupervised clustering.

### 3.1 Basic GMM Fitting

```python
import torch
import numpy as np
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import Normal

# Simulate two-cluster 1D data
torch.manual_seed(42)
cluster1 = torch.normal(mean=2.0, std=0.5, size=(200,))
cluster2 = torch.normal(mean=7.0, std=1.0, size=(200,))
data = torch.cat([cluster1, cluster2]).unsqueeze(1)

# Define GMM with 2 Normal components
gmm = GeneralMixtureModel([Normal(), Normal()])
gmm.fit(data)

print("Means:", [c.means.item() for c in gmm.distributions])
print("Weights:", gmm.priors)
```

### 3.2 Multivariate GMM for Cell Clustering

```python
import torch
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import MultivariateGaussian

# Simulate scRNA-seq-like data: 3 cell types, 4 gene features
torch.manual_seed(0)
cell_type1 = torch.randn(150, 4) + torch.tensor([2.0, 0.0, -1.0, 1.5])
cell_type2 = torch.randn(150, 4) + torch.tensor([-2.0, 3.0, 0.5, -1.0])
cell_type3 = torch.randn(150, 4) + torch.tensor([0.0, -2.0, 3.0, 0.0])
data = torch.cat([cell_type1, cell_type2, cell_type3])

# Fit GMM with 3 multivariate Gaussian components
gmm = GeneralMixtureModel([
    MultivariateGaussian(),
    MultivariateGaussian(),
    MultivariateGaussian()
])
gmm.fit(data)

# Predict cluster assignments (soft assignments)
log_probs = gmm.predict_proba(data)
hard_labels = gmm.predict(data)

print("Unique labels:", torch.unique(hard_labels))
print("Component means:")
for i, dist in enumerate(gmm.distributions):
    print(f"  Cluster {i}: {dist.means.numpy().round(2)}")
```

### 3.3 Initializing with Prior Knowledge

```python
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import Normal

# You know roughly where clusters are — seed them
gmm = GeneralMixtureModel([
    Normal([1.0], [0.5]),   # Prior: cluster 1 near 1.0
    Normal([6.0], [1.0]),   # Prior: cluster 2 near 6.0
    Normal([12.0], [1.5])   # Prior: cluster 3 near 12.0
], priors=[0.4, 0.3, 0.3])  # Unequal cluster sizes

# Freeze priors so they don't update during EM
gmm.freeze_distributions()   # optional — keeps components fixed
```

---

## 4. Hidden Markov Models (HMM)

HMMs are sequential models where observations depend on a *hidden* (latent) state sequence. If you're doing sequence analysis — HMMs are your bread and butter.

**Key stat:** HMMs underpin ~60% of classical gene finding tools (GENSCAN, Augustus, etc.) and are foundational in protein family databases like Pfam.

### 4.1 Dense HMM — Basic Setup

```python
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

# Model a 3-state HMM: low/medium/high expression states
model = DenseHMM([
    Normal([1.0], [0.5]),   # State 0: low expression
    Normal([5.0], [1.0]),   # State 1: medium expression
    Normal([10.0], [1.5])   # State 2: high expression
])

# Transition matrix: [from_state x to_state]
edges = torch.log(torch.tensor([
    [0.7, 0.2, 0.1],   # from state 0
    [0.1, 0.7, 0.2],   # from state 1
    [0.1, 0.2, 0.7]    # from state 2
]))

# Starting probabilities
starts = torch.log(torch.tensor([0.6, 0.3, 0.1]))
ends   = torch.log(torch.tensor([0.3, 0.4, 0.3]))

model.edges = edges
model.starts = starts
model.ends = ends
```

### 4.2 Training an HMM on Sequence Data

```python
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Normal

torch.manual_seed(7)

# Simulate sequences from 2 hidden states
def simulate_hmm_data(n_seqs=50, seq_len=30):
    seqs = []
    for _ in range(n_seqs):
        state = 0
        seq = []
        for _ in range(seq_len):
            if state == 0:
                obs = torch.normal(2.0, 0.5, (1,))
                state = 0 if torch.rand(1) < 0.8 else 1
            else:
                obs = torch.normal(8.0, 1.0, (1,))
                state = 1 if torch.rand(1) < 0.75 else 0
            seq.append(obs)
        seqs.append(torch.stack(seq))
    return seqs

sequences = simulate_hmm_data()  # list of (30, 1) tensors

# Build and fit model
model = DenseHMM([Normal(), Normal()], max_iter=100, tol=1e-4, verbose=True)
model.fit(sequences)

print("Learned means:", [d.means.item() for d in model.distributions])
print("Transition matrix:", torch.exp(model.edges))
```

### 4.3 Viterbi Decoding — Most Likely State Path

```python
# Given a trained model, decode the most probable state sequence
test_seq = sequences[0]  # shape: (30, 1)

# predict returns the most likely state at each timestep
state_path = model.predict(test_seq.unsqueeze(0))  # shape: (1, 30)
print("Viterbi path:", state_path[0])

# Forward algorithm: total sequence log-likelihood
log_likelihood = model.log_probability(test_seq.unsqueeze(0))
print("Log-likelihood:", log_likelihood.item())
```

### 4.4 DNA Motif Finder with HMM

A real-world use case — detecting a known motif embedded in background sequence.

```python
import torch
from pomegranate.hmm import DenseHMM
from pomegranate.distributions import Categorical

# Encoding: A=0, C=1, G=2, T=3
# 2-state model: background vs motif
# Background: uniform distribution
# Motif: biased toward specific bases (e.g. TATA box: T-A-T-A)

background = Categorical([[0.25, 0.25, 0.25, 0.25]])

# TATA box consensus positions
tata_T = Categorical([[0.02, 0.02, 0.02, 0.94]])  # strongly T
tata_A = Categorical([[0.94, 0.02, 0.02, 0.02]])  # strongly A

# For a simple demo, 2 states: background and motif
model = DenseHMM([background, tata_T], max_iter=200)

# In practice: build a linear chain of motif states, one per position
# This gives you a profile HMM — the same architecture used by HMMER
```

### 4.5 Sparse HMM for Large State Spaces

When transitions are sparse (not all states connect to all others):

```python
from pomegranate.hmm import SparseHMM
from pomegranate.distributions import Normal

# Useful when you have many states but sparse connectivity
# e.g. gene structure models with exon/intron/UTR states

distributions = [Normal([i * 2.0], [0.8]) for i in range(10)]  # 10 states
model = SparseHMM(distributions)

# SparseHMM is more memory-efficient than DenseHMM for large state spaces
# DenseHMM: O(N²) transition matrix
# SparseHMM: O(E) where E = number of actual edges
```

---

## 5. Bayesian Networks

A Bayesian Network is a DAG (Directed Acyclic Graph) where nodes are random variables and edges encode conditional dependencies.

### 5.1 Simple 3-Node Network

```python
import torch
from pomegranate.bayesian_network import BayesianNetwork

# Classic example: Rain → Sprinkler → Wet Grass
# All variables are binary: 0=False, 1=True

# Data columns: [Rain, Sprinkler, WetGrass]
torch.manual_seed(42)
n = 1000

rain       = torch.bernoulli(torch.full((n,), 0.3))
sprinkler  = torch.bernoulli(0.4 * (1 - rain) + 0.01 * rain)
wet        = torch.bernoulli(0.9 * sprinkler + 0.8 * rain - 0.7 * sprinkler * rain + 0.0 * (1 - sprinkler) * (1 - rain))

data = torch.stack([rain, sprinkler, wet], dim=1).int()

# Learn structure and parameters from data
model = BayesianNetwork()
model.fit(data)

print("Learned structure:")
print(model.structure)
```

### 5.2 Inference — Querying the Network

```python
# Given Wet Grass = True, what's the probability it rained?
query = torch.tensor([[float('nan'), float('nan'), 1.0]])

# marginal inference
probs = model.predict_proba(query)
print("P(Rain | WetGrass=True):", probs[0][0])
print("P(Sprinkler | WetGrass=True):", probs[0][1])
```

### 5.3 Gene Regulatory Network Toy Example

```python
import torch
from pomegranate.bayesian_network import BayesianNetwork

# Toy gene regulation:
# TF_A activates Gene_B
# TF_A + Gene_B co-regulate Gene_C
# Columns: [TF_A, Gene_B, Gene_C]

n = 2000
tf_a   = torch.bernoulli(torch.full((n,), 0.5))
gene_b = torch.bernoulli(0.8 * tf_a + 0.1 * (1 - tf_a))
gene_c = torch.bernoulli(0.9 * gene_b * tf_a + 0.2 * (1 - gene_b))

data = torch.stack([tf_a, gene_b, gene_c], dim=1).int()

bn = BayesianNetwork(algorithm='greedy')
bn.fit(data)

print("Inferred network structure:", bn.structure)
# Ideally: ((), (0,), (0, 1)) — Gene_C depends on TF_A and Gene_B
```

---

## 6. Naive Bayes Classifier

Naive Bayes assumes feature independence given the class — naive, yes, but surprisingly effective, especially in high dimensions.

**Research backing:** Naive Bayes classifiers achieve ~80–95% accuracy on text classification tasks and remain competitive on genomic feature classification despite their simplicity (McCallum & Nigam, 1998).

### 6.1 Basic Classification

```python
import torch
from pomegranate.naive_bayes import NaiveBayes
from pomegranate.distributions import Normal

# Binary classification with 4 continuous features
# Imagine: normal vs tumor samples based on 4 gene expression values
torch.manual_seed(0)

normal_data = torch.randn(200, 4) + torch.tensor([1.0, -1.0, 0.5, 2.0])
tumor_data  = torch.randn(200, 4) + torch.tensor([-1.0, 2.0, -0.5, -1.0])

X = torch.cat([normal_data, tumor_data])
y = torch.cat([torch.zeros(200), torch.ones(200)]).int()

# Fit Naive Bayes
nb = NaiveBayes([
    Normal(),  # Class 0 (normal)
    Normal()   # Class 1 (tumor)
])
nb.fit(X, y)

# Predict
preds = nb.predict(X)
acc = (preds == y).float().mean()
print(f"Accuracy: {acc:.4f}")
```

### 6.2 Mixed-Distribution Naive Bayes

When features come from different distributions (a real-world scenario you'll encounter):

```python
from pomegranate.naive_bayes import NaiveBayes
from pomegranate.distributions import IndependentComponents, Normal, Poisson

# Feature layout:
#   cols 0–1: continuous gene expression values (Normal)
#   col 2: read count (Poisson)

class0_dist = IndependentComponents([Normal([0.0], [1.0]),
                                     Normal([0.0], [1.0]),
                                     Poisson([5.0])])

class1_dist = IndependentComponents([Normal([3.0], [1.0]),
                                     Normal([-2.0], [1.0]),
                                     Poisson([15.0])])

nb = NaiveBayes([class0_dist, class1_dist])

# Simulate mixed data
import torch
n = 300
X = torch.cat([
    torch.randn(n, 2),
    torch.poisson(torch.full((n, 1), 5.0))
], dim=1)
y = torch.zeros(n).int()

nb.fit(X, y)  # Fit on class 0 only — then fit full dataset with both classes
```

---

## 7. Markov Chains

A Markov Chain models sequential data where the next state depends *only* on the current state. The "memoryless" property.

### 7.1 First-Order Markov Chain

```python
import torch
from pomegranate.markov_chain import MarkovChain

# DNA sequence modeling: A=0, C=1, G=2, T=3
# Fit a first-order Markov chain to capture dinucleotide preferences

sequences = [
    torch.tensor([0, 1, 2, 3, 0, 1, 2, 3, 0, 2], dtype=torch.int32),
    torch.tensor([1, 1, 0, 2, 2, 3, 1, 0, 0, 1], dtype=torch.int32),
    torch.tensor([3, 0, 1, 1, 2, 0, 3, 3, 1, 2], dtype=torch.int32),
]

mc = MarkovChain(k=1)  # k=1: first-order
mc.fit(sequences)

# Transition probabilities (4x4 matrix)
print("Transition matrix:")
print(torch.exp(mc.distributions[1].probs))
```

### 7.2 Higher-Order Markov Chain for Sequence Modeling

```python
from pomegranate.markov_chain import MarkovChain

# k=3: uses trigram context (good for codon-level modeling)
mc3 = MarkovChain(k=3)
mc3.fit(sequences)

# Log probability of a new sequence
test_seq = torch.tensor([0, 1, 2, 3, 0, 2, 1], dtype=torch.int32)
log_p = mc3.log_probability(test_seq.unsqueeze(0))
print("Log P(sequence):", log_p.item())
```

### 7.3 Protein Secondary Structure Markov Model

```python
import torch
from pomegranate.markov_chain import MarkovChain

# Secondary structure states: H=Helix(0), E=Sheet(1), C=Coil(2)
# Sequences of secondary structure annotations
ss_sequences = [
    torch.tensor([0, 0, 0, 2, 1, 1, 2, 0, 0], dtype=torch.int32),
    torch.tensor([1, 1, 2, 0, 0, 0, 2, 1], dtype=torch.int32),
    torch.tensor([2, 0, 0, 2, 2, 1, 1, 1, 0], dtype=torch.int32),
]

mc = MarkovChain(k=2)
mc.fit(ss_sequences)

# P(Coil → Helix → Helix transition)
test = torch.tensor([2, 0, 0], dtype=torch.int32)
lp = mc.log_probability(test.unsqueeze(0))
print("Log P([Coil, Helix, Helix]):", lp.item())
```

---

## 8. Factor Graphs

Factor graphs are a general framework for representing factored joint distributions. They generalize Bayesian networks and Markov Random Fields.

```python
# Factor graphs in pomegranate work through the FactorGraph class
# They're particularly powerful for belief propagation

# Note: Factor graph support varies by pomegranate version.
# The DenseHMM and BayesianNetwork internally use factor graph-like inference.

# For custom factor graphs, you compose distributions and specify
# the dependency structure manually. This is an advanced use case —
# most bioinformatics applications are served by HMMs or Bayesian networks.
```

---

## 9. Handling Missing Data

This is where pomegranate genuinely shines. Most libraries pretend missing data doesn't exist. Pomegranate treats `NaN` as unobserved — fitting and inference happen over the *marginal* distribution.

### 9.1 Fitting with Missing Values

```python
import torch
from pomegranate.distributions import MultivariateGaussian

# Simulate missing data (realistic in genomics — censored measurements)
data = torch.tensor([
    [1.2, 2.3, float('nan')],
    [float('nan'), 1.8, 3.1],
    [0.9, float('nan'), 2.7],
    [1.5, 2.1, 2.9],
    [1.1, float('nan'), float('nan')],
], dtype=torch.float32)

mvg = MultivariateGaussian()
mvg.fit(data)  # Internally uses EM to handle NaNs

print("Mean (fitted with missing values):", mvg.means)
```

### 9.2 Imputation / Prediction

```python
from pomegranate.distributions import Normal

dist = Normal()
complete_data = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]], dtype=torch.float32)
dist.fit(complete_data)

# Impute missing value by computing expected value under fitted distribution
# (mean imputation using the fitted distribution)
print("Fitted mean:", dist.means.item())   # best estimate for any missing value
print("Fitted std:", dist.covs.item())
```

### 9.3 Missing Data in GMMs

```python
import torch
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.distributions import MultivariateGaussian

# 3D data with ~20% missing at random
torch.manual_seed(1)
n = 300
data = torch.randn(n, 3)

# Randomly zero out ~20% of entries
mask = torch.rand(n, 3) < 0.2
data[mask] = float('nan')

gmm = GeneralMixtureModel([MultivariateGaussian(), MultivariateGaussian()])
gmm.fit(data)  # EM handles NaNs gracefully — no imputation needed upfront

print("GMM fitted successfully despite missing data")
print("Component means:")
for i, d in enumerate(gmm.distributions
