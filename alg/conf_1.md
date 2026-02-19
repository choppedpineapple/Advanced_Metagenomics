# Hidden Markov Models — Step by Step with `hmmlearn`

Let's use the classic **"Weather → Activity"** HMM. Simple enough to understand, just complex enough to be meaningful.

**The Setup:**
- 🌦️ **Hidden states** (what we can't observe): `Rain`, `Sunny`
- 👀 **Observations** (what we can see): `Walk`, `Shop`, `Clean`

The whole point of HMMs: *you see the activities, you want to infer the weather.*

---

## Step 0 — Install & Import

```python
pip install hmmlearn numpy
```

```python
import numpy as np
from hmmlearn import hmm
import warnings
warnings.filterwarnings("ignore")

# Seed for reproducibility
np.random.seed(42)
```

**Why this matters:** `hmmlearn` expects numpy arrays everywhere. No pandas, no lists — numpy arrays. Burn that into your brain now so you don't suffer later.

---

## Step 1 — Define the 3 Core Probability Matrices

These are the heart of any HMM. Every HMM is fully described by **π, A, B**.

```python
# -----------------------------------------------
# π — Initial State Probabilities
# "What's the probability the first day is Rainy vs Sunny?"
# Shape: (n_states,)
# -----------------------------------------------
initial_probs = np.array([0.6, 0.4])
#                          ^     ^
#                        Rain  Sunny
# States order: 0=Rain, 1=Sunny (we define this — hmmlearn doesn't know names)

print("Initial probabilities (π):")
print(f"  Rain:  {initial_probs[0]}")
print(f"  Sunny: {initial_probs[1]}")
print(f"  Sum:   {initial_probs.sum()}  ← must always be 1.0\n")
```

**Output:**
```
Initial probabilities (π):
  Rain:  0.6
  Sunny: 0.4
  Sum:   1.0  ← must always be 1.0
```

---

```python
# -----------------------------------------------
# A — Transition Matrix
# "Given I'm in state X today, what's the prob of state Y tomorrow?"
# Shape: (n_states, n_states)
# Rows = FROM state, Columns = TO state
# -----------------------------------------------
transition_matrix = np.array([
    [0.7, 0.3],   # From Rain  → [Stay Rain, Go Sunny]
    [0.4, 0.6]    # From Sunny → [Go Rain,   Stay Sunny]
])

print("Transition Matrix (A):")
print("         → Rain  → Sunny")
print(f"Rain   :  {transition_matrix[0]}")
print(f"Sunny  :  {transition_matrix[1]}")
print(f"\nRow sums: {transition_matrix.sum(axis=1)}  ← each row must sum to 1\n")
```

**Output:**
```
Transition Matrix (A):
         → Rain  → Sunny
Rain   :  [0.7 0.3]
Sunny  :  [0.4 0.6]

Row sums: [1. 1.]  ← each row must sum to 1
```

> 💡 **Intuition:** Rain is "sticky" (70% chance it stays rainy). Sunny days are less sticky (60% chance it stays sunny). This mirrors real Irish weather almost perfectly.

---

```python
# -----------------------------------------------
# B — Emission Matrix
# "Given I'm in hidden state X, what obs do I produce?"
# Shape: (n_states, n_observations)
# Rows = states, Columns = observation symbols
# -----------------------------------------------
emission_matrix = np.array([
    [0.1, 0.4, 0.5],   # Rain  emits: [Walk, Shop, Clean]
    [0.6, 0.3, 0.1]    # Sunny emits: [Walk, Shop, Clean]
])
#                         Walk  Shop  Clean

print("Emission Matrix (B):")
print("          Walk  Shop  Clean")
print(f"Rain  :  {emission_matrix[0]}")
print(f"Sunny :  {emission_matrix[1]}")
print(f"\nRow sums: {emission_matrix.sum(axis=1)}  ← must each sum to 1\n")
```

**Output:**
```
Emission Matrix (B):
          Walk  Shop  Clean
Rain  :  [0.1 0.4 0.5]
Sunny :  [0.6 0.3 0.1]

Row sums: [1. 1.]  ← must each sum to 1
```

> 💡 **Intuition:** When it rains, you mostly clean or shop indoors. When it's sunny, you walk. Makes sense.

---

**Before Step 2 — What do we have so far?**

| Matrix | Shape | Meaning |
|--------|-------|---------|
| π | `(2,)` | Starting state distribution |
| A | `(2, 2)` | State-to-state transition probs |
| B | `(2, 3)` | State-to-observation emission probs |

Three matrices. That's literally your entire model. Everything else is math on top of these.

---

## Step 2 — Build the HMM Object

```python
# CategoricalHMM = discrete observations (our case: Walk/Shop/Clean)
# n_components = number of hidden states
model = hmm.CategoricalHMM(n_components=2, n_iter=100)

# Manually plug in our matrices instead of letting hmmlearn randomly init them
model.startprob_ = initial_probs       # π
model.transmat_  = transition_matrix   # A
model.emissionprob_ = emission_matrix  # B

print("Model built!")
print(f"  Hidden states : {model.n_components}")
print(f"  Observation symbols: {model.n_features}")
```

**Output:**
```
Model built!
  Hidden states : 2
  Observation symbols: 3
```

> Notice `.startprob_`, `.transmat_`, `.emissionprob_` — the trailing underscore is a sklearn/hmmlearn convention meaning "this attribute has been set/fitted."

---

## Step 3 — Generate Sequences (Sample from the Model)

This answers: *"If this HMM is true, what data would it produce?"*

```python
# sample() returns:
#   obs_seq   — the observations you'd actually see   (Walk/Shop/Clean as 0/1/2)
#   state_seq — the TRUE hidden states that generated them (Rain/Sunny as 0/1)
#                ↑ In real life you'd NEVER see this. This is the "hidden" part.

obs_seq, state_seq = model.sample(n_samples=20)

# Map integers back to readable labels
state_map = {0: "Rain ", 1: "Sunny"}
obs_map   = {0: "Walk ", 1: "Shop ", 2: "Clean"}

print("Generated sequence (20 time steps):\n")
print("t   | Hidden State | Observation")
print("----|--------------|------------")
for t, (s, o) in enumerate(zip(state_seq, obs_seq)):
    print(f"t={t:<2} | {state_map[s]}       | {obs_map[o[0]]}")
```

**Example Output:**
```
t   | Hidden State | Observation
----|--------------|------------
t=0 | Rain         | Clean
t=1 | Rain         | Shop
t=2 | Rain         | Clean
t=3 | Sunny        | Walk
t=4 | Sunny        | Walk
t=5 | Rain         | Shop
t=6 | Rain         | Clean
...
```

> See how **Rain days cluster together** (because A[Rain→Rain]=0.7) and **Sunny days cluster too**? That's the Markov property doing its thing. It's not random noise — there's temporal structure baked in.

---

## Step 4 — Decoding: Viterbi Algorithm

Now flip it around. **You only see the observations. Infer the hidden states.**

This is the actual use-case. In genomics, for example, you observe GC-content and want to infer CpG islands (hidden states).

```python
# obs_seq is shape (n_samples, 1) — hmmlearn wants it this way
# lengths tells hmmlearn this is one sequence of length 20
lengths = [len(obs_seq)]

# Viterbi decoding: finds the MOST PROBABLE hidden state sequence
log_prob, predicted_states = model.decode(obs_seq, lengths=lengths, algorithm="viterbi")

print(f"Log probability of sequence: {log_prob:.4f}\n")
print("Comparison: True vs Predicted hidden states\n")
print("t   | TRUE State   | PREDICTED    | Match?")
print("----|--------------|--------------|-------")
correct = 0
for t, (true, pred) in enumerate(zip(state_seq, predicted_states)):
    match = "✓" if true == pred else "✗"
    if true == pred: correct += 1
    print(f"t={t:<2} | {state_map[true]}      | {state_map[pred]}      | {match}")

print(f"\nAccuracy: {correct}/{len(state_seq)} = {correct/len(state_seq)*100:.1f}%")
```

**Example Output:**
```
Accuracy: 17/20 = 85.0%
```

> 85% accuracy just from three probability matrices and a smart algorithm. No neural network, no GPU, no 80GB model. Elegant? Yes. Embarrassingly so.

---

## Step 5 — The Forward Algorithm (Scoring)

Sometimes you don't want the best path, you want: *"How likely is this sequence under this model?"*

```python
# score() computes log P(observations | model) using the Forward algorithm
log_likelihood = model.score(obs_seq, lengths)

print(f"Log-likelihood of observed sequence: {log_likelihood:.4f}")
print(f"Likelihood (actual prob):            {np.exp(log_likelihood):.6e}")
print()
print("Why log? Because multiplying 20 tiny probabilities → numerical underflow.")
print("log(0.001 × 0.002 × ...) is much safer than 0.001 × 0.002 × ...")
```

**Output:**
```
Log-likelihood of observed sequence: -21.3842
Likelihood (actual prob):             5.21e-10
```

> That tiny actual probability is why we always work in log-space. If you multiply 20 numbers each around 0.3 together, Python will eventually hand you `0.0` and your model will silently lie to your face.

---

## Full Summary — What Each Matrix Does

```
π  (startprob_)     → Where does the chain START?
A  (transmat_)      → How does the chain MOVE through states?
B  (emissionprob_)  → What does each state PRODUCE?

Together: π + A + B = a complete generative story for your sequence data.

Algorithms on top:
  Forward  → P(obs | model)              [scoring]
  Viterbi  → argmax P(states | obs, model) [decoding]
  Baum-Welch → learn π, A, B from data   [training — hmmlearn's .fit()]
```

---

## Bonus — One-liner to Train (Baum-Welch)

If you don't know the matrices and want to *learn* them from data:

```python
# Generate training data first (pretend we don't know the true model)
train_obs, _ = model.sample(n_samples=500)

# Create a new untrained model
learner = hmm.CategoricalHMM(n_components=2, n_iter=100)

# Baum-Welch (EM algorithm) learns π, A, B from raw observations
learner.fit(train_obs, lengths=[500])

print("Learned transition matrix:")
print(np.round(learner.transmat_, 3))
print("\nTrue transition matrix:")
print(transition_matrix)
```

You'll see the learned matrices are *close* to the true ones — the model figures out the structure just from watching activities, never seeing the weather. That's the magic of HMMs.

---

The key insight to remember: **you're always reasoning about something you can't see, using something you can**. That's it. Everything else is just linear algebra wrapped in clever dynamic programming.
