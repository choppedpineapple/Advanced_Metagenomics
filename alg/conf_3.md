To understand Hidden Markov Models (HMMs) effectively, it helps to frame the problem with a story.

**The Scenario:** Imagine we are observing a person in a windowless room.
1.  **Hidden States:** We cannot see the weather outside (Sunny or Rainy). These are the "Hidden" states.
2.  **Observations:** We can only see what the person is wearing (a T-shirt or a Hoodie). These are the "Emissions".

We will use the `hmmlearn` library because it integrates smoothly with Python's numerical library, NumPy.

### Prerequisites
You will need to install the library first:
```bash
pip install hmmlearn numpy
```

---

### Step 1: Define the Raw Data (Observations)
Before we build the model, we need data. Since HMMs deal with sequences, the order matters.

**Goal:** Convert real-world observations into numbers the computer understands.

**Code:**
```python
import numpy as np
from hmmlearn import hmm

# 1. Define a mapping for our observations
# 0 = T-shirt
# 1 = Hoodie
observation_map = {0: 'T-shirt', 1: 'Hoodie'}

# 2. Create a sequence of observations (what we saw)
# Raw Data: [T-shirt, T-shirt, Hoodie, T-shirt, Hoodie, Hoodie]
raw_observations = [0, 0, 1, 0, 1, 1]

# 3. Reshape data for hmmlearn
# hmmlearn expects a 2D array: (Number_of_samples, Feature_Dimension)
observations = np.array(raw_observations).reshape(-1, 1)

print("Original List:", raw_observations)
print("Model Input Format:\n", observations)
```

**What happened?**
*   **Before:** We had a simple Python list `[0, 0, 1...]`.
*   **After:** We have a NumPy column vector. This format is required because HMMs can theoretically handle multiple features at once (e.g., clothing *and* temperature), so it expects columns even if we only have one.

---

### Step 2: Initialize the Model Structure
Now we create the "container" for our probabilities.

**Goal:** Tell the model how many hidden states and observation types exist.

**Code:**
```python
# 1. Define model parameters
n_states = 2       # We suspect there are 2 hidden weather patterns (Sunny, Rainy)
n_observations = 2 # We know there are 2 types of clothing (T-shirt, Hoodie)

# 2. Initialize the MultinomialHMM model
# 'n_iter' is how many times it will refine its guess during training
model = hmm.MultinomialHMM(n_components=n_states, n_iter=0, tol=0.01)

# Note: We set n_iter=0 initially because we are going to MANUALLY set the 
# probabilities first to understand the matrices. Later we will train it.
```

**What happened?**
*   **Before:** An empty idea of a model.
*   **After:** A model object is created. It knows it needs to find 2 states, but it currently has random (useless) numbers inside it.

---

### Step 3: Manually "Seed" the Matrices (The Concept Phase)
This is the most critical step for understanding. We will manually inject our "beliefs" about the world into the model. This creates the three matrices you asked for.

**Goal:** Define Initial ($\pi$), Transition ($A$), and Emission ($B$) matrices.

**Code:**
```python
# --- Matrix 1: Initial Probability (Start Probability) ---
# Probability of the weather being Sunny or Rainy on Day 1.
# Let's assume it's usually Sunny at the start.
# [P(Sunny), P(Rainy)]
initial_prob = np.array([0.8, 0.2])

# --- Matrix 2: Transition Probability ---
# Probability of moving from one weather state to another.
# Row 1: From Sunny -> [To Sunny, To Rainy]
# Row 2: From Rainy -> [To Sunny, To Rainy]
transition_prob = np.array([
    [0.7, 0.3], # If today is Sunny, 70% chance tomorrow is Sunny
    [0.4, 0.6]  # If today is Rainy, 40% chance tomorrow is Sunny
])

# --- Matrix 3: Emission Probability ---
# Probability of seeing a clothing type given the weather.
# Row 1: If Sunny -> [P(T-shirt), P(Hoodie)]
# Row 2: If Rainy -> [P(T-shirt), P(Hoodie)]
emission_prob = np.array([
    [0.9, 0.1], # If Sunny, very likely to wear T-shirt
    [0.2, 0.8]  # If Rainy, very likely to wear Hoodie
])

# Inject these into the model object
model.startprob_ = initial_prob
model.transmat_ = transition_prob
model.emissionprob_ = emission_prob

print("Initial Probabilities (Pi):\n", model.startprob_)
print("\nTransition Matrix (A):\n", model.transmat_)
print("\nEmission Matrix (B):\n", model.emissionprob_)
```

**What happened?**
*   **Before:** The model was empty/dumb.
*   **After:** The model now has "logic." It knows that Sunny days usually follow Sunny days, and Hoodies are usually worn when it's Rainy.

---

### Step 4: Using the Model (The Inference Phase)
Now that the model contains our matrices, we can ask it questions. We will ask it to "decode" the hidden states based on the observations we created in Step 1.

**Goal:** Use the matrices to guess the weather.

**Code:**
```python
# We want to predict the hidden states (weather) for our observations (clothing)
# The .predict method uses the Viterbi algorithm internally.
log_prob, hidden_states = model.decode(observations, algorithm="viterbi")

print("Observations (Clothing):", [observation_map[o[0]] for o in observations])
print("Predicted Hidden States (Weather IDs):", hidden_states)

# Map state IDs to names for clarity
# (Note: HMM assigns IDs 0 and 1 arbitrarily during manual setup. 
# Based on our probabilities, State 0 is likely Sunny and State 1 is Rainy)
state_map = {0: 'Sunny', 1: 'Rainy'}
predicted_weather = [state_map[s] for s in hidden_states]

print("Predicted Weather Sequence:", predicted_weather)
```

**What happened?**
*   **Before:** We had a sequence of clothing: `[T-shirt, T-shirt, Hoodie...]`.
*   **After:** The model calculated the most likely path of hidden states: `[Sunny, Sunny, Rainy...]`.
*   **Connection:** The model looked at the **Emission Matrix** to see what weather produces T-shirts, and the **Transition Matrix** to see if it's plausible for the weather to switch from Sunny to Rainy between observations.

---

### Step 5: "Learning" (Generating Matrices from Data)
In Step 3, we *gave* the model the matrices. But usually, you want the model to *calculate* the matrices for you (which seems to be your ultimate goal).

**Goal:** Reset the model, feed it data, and watch it generate the matrices itself.

**Code:**
```python
print("\n--- LEARNING PHASE ---")

# 1. Create a new, "blank" model
# We tell it n_components=2 (looking for 2 hidden states)
# n_iter=100 means it will refine its guesses 100 times.
learn_model = hmm.MultinomialHMM(n_components=2, n_iter=100, random_state=42)

# 2. Train the model
# We feed it the observations. It doesn't know the weather; it has to guess the structure.
learn_model.fit(observations)

# 3. Extract the learned matrices
print("\nLearned Initial Probabilities:\n", learn_model.startprob_)
print("\nLearned Transition Matrix:\n", learn_model.transmat_)
print("\nLearned Emission Matrix:\n", learn_model.emissionprob_)
```

**What happened?**
*   **Before:** The model started with random numbers for the matrices.
*   **After:** The model adjusted those numbers iteratively (using the Baum-Welch algorithm) to maximize the probability of generating the observation sequence we gave it.
*   **Note on Results:** Since our data sample was very small (only 6 items), the learned matrices might look different from our manual ones. If you provide a massive dataset of observations, the learned matrices will converge to the "truth."

### Summary of Matrices

1.  **Initial Probability ($\pi$):**
    *   **Question:** Where do we start?
    *   **Shape:** (1 x Number of States)
    *   **Example:** `[0.8, 0.2]` -> 80% chance we start in State 0.

2.  **Transition Probability ($A$):**
    *   **Question:** How do we move between hidden states?
    *   **Shape:** (States x States)
    *   **Example:** `matrix[0][1]` is the probability of moving from State 0 to State 1.

3.  **Emission Probability ($B$):**
    *   **Question:** What do we observe given a hidden state?
    *   **Shape:** (States x Observations)
    *   **Example:** `matrix[0][1]` is the probability of observing Output 1 while in State 0.
