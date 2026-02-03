### Block 1: Setup
Just like any other PyTorch project, we start with the basics. In Bioinformatics, we usually process massive datasets, so having a GPU is very helpful, but CPU works for learning.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Check if we have a GPU (crucial for processing large genomic datasets)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Block 2: Encoding DNA (Letters to Numbers)
Computers cannot understand letters like 'A', 'C', 'G', 'T'. In Bioinformatics, the first step is usually **One-Hot Encoding**.
*   **A** becomes `[1, 0, 0, 0]`
*   **C** becomes `[0, 1, 0, 0]`
*   **G** becomes `[0, 0, 1, 0]`
*   **T** becomes `[0, 0, 0, 1]`

```python
# A simple dictionary to map DNA letters to numbers
dna_map = {'A': [1., 0., 0., 0.], 
           'C': [0., 1., 0., 0.], 
           'G': [0., 0., 1., 0.], 
           'T': [0., 0., 0., 1.]}

# Example DNA Sequence
sequence = "ACGT"

# Convert string to a list of vectors
encoded_seq = [dna_map[base] for base in sequence]

# Convert to PyTorch Tensor
# Shape: (Sequence_Length, 4 channels)
dna_tensor = torch.tensor(encoded_seq)

print(f"Original: {sequence}")
print(f"Tensor shape: {dna_tensor.shape}") 
# Output: (4, 4) -> 4 bases, 4 channels each
```

### Block 3: Simulating Gene Expression Data
Instead of loading a massive CSV file (which takes time), let's simulate a common bioinformatics task: **Predicting Disease from Gene Expression.**

*   **Scenario:** We have 100 patients.
*   **Features:** We measured the expression levels of 50 genes for each patient.
*   **Labels:** `1` = Has Disease, `0` = Healthy.

```python
import numpy as np

# Simulate Data
# 100 patients, 50 gene expression values each (normalized between 0 and 1)
X = np.random.rand(100, 50) 

# Labels: Randomly assign 0 (Healthy) or 1 (Sick)
y = np.random.randint(0, 2, 100)

# Convert to PyTorch Tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1) # Reshape for calculations

print(f"Gene Expression Data Shape: {X_tensor.shape}") # 100 Patients, 50 Genes
print(f"Labels Shape: {y_tensor.shape}")
```

### Block 4: Building a Simple "Gene Predictor" (Feed Forward)
This is a standard neural network, but we call it a "Gene Predictor." It takes the gene expression levels and outputs a probability of disease.

```python
class GenePredictor(nn.Module):
    def __init__(self, input_genes):
        super(GenePredictor, self).__init__()
        # Layer 1: Connects 50 genes to 32 hidden neurons
        self.fc1 = nn.Linear(input_genes, 32)
        # Layer 2: Connects 32 hidden neurons to 1 output (Probability)
        self.fc2 = nn.Linear(32, 1)
        # Dropout: Helps prevent overfitting on small medical datasets
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Pass through first layer and apply ReLU activation
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        # Pass through output layer
        # We use Sigmoid at the end to get a probability between 0 and 1
        x = torch.sigmoid(self.fc2(x))
        return x

# Initialize model (Input is 50 genes)
model = GenePredictor(input_genes=50).to(device)
print(model)
```

### Block 5: Analyzing DNA Sequences (1D Convolution)
For DNA sequences (text-like data), we use **1D Convolutions**.
*   **Concept:** Imagine sliding a small window across the DNA to find specific "motifs" or patterns (like "TATA Box").

```python
class DNA_Motif_Finder(nn.Module):
    def __init__(self):
        super(DNA_Motif_Finder, self).__init__()
        
        # 1D Convolution
        # in_channels=4 (A,C,G,T), out_channels=16 (filters), kernel_size=3 (look at 3 bases at a time)
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3)
        
        # Fully connected layer to make the final decision
        self.fc = nn.Linear(16, 1) 

    def forward(self, x):
        # Input x shape: (Batch, Channels, Length)
        # Convolution slides across the length dimension
        x = self.conv1(x)
        x = torch.relu(x)
        
        # Global Max Pooling: Find the strongest motif match across the whole sequence
        x = torch.max(x, dim=2)[0]
        
        # Output probability
        x = torch.sigmoid(self.fc(x))
        return x

# Dummy DNA data: Batch of 10 sequences, 4 channels, 50 base pairs long
dummy_dna = torch.randn(10, 4, 50).to(device)
dna_model = DNA_Motif_Finder().to(device)

output = dna_model(dummy_dna)
print(f"DNA Model Output Shape: {output.shape}")
```

### Block 6: Drug Discovery (Molecular Fingerprints)
In drug discovery, molecules are often represented as **binary vectors (Fingerprints)**.
*   **Scenario:** You have a vector of size 1024. A `1` means a specific substructure exists in the molecule; `0` means it doesn't.
*   **Goal:** Predict if the molecule is toxic or soluble.

```python
class DrugToxicityPredictor(nn.Module):
    def __init__(self, fingerprint_size):
        super(DrugToxicityPredictor, self).__init__()
        # Input: 1024 fingerprint bits -> Hidden Layer
        self.hidden = nn.Linear(fingerprint_size, 256)
        # Hidden -> Output (Toxicity Score)
        self.output = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.relu(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x

# Simulate Molecular Fingerprints (1024 bits)
# Batch of 5 molecules
drug_data = torch.randint(0, 2, (5, 1024)).float().to(device) 

drug_model = DrugToxicityPredictor(1024).to(device)
toxicity_scores = drug_model(drug_data)

print(f"Toxicity Scores (0=Safe, 1=Toxic):\n{toxicity_scores}")
```

### Block 7: Training the Gene Model (Simple Loop)
Here is the training loop. In Bioinformatics, datasets are often small, so we usually train for more epochs with a smaller learning rate to be careful not to overfit.

```python
# Setup Data for training
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Loss Function: Binary Cross Entropy (Standard for Yes/No classification)
criterion = nn.BCELoss() 
# Optimizer: Adam (Good for biological data convergence)
optimizer = optim.Adam(model.parameters(), lr=0.001) 

# Training Loop
print("Training Gene Expression Model...")
for epoch in range(10): # 10 Epochs
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        
        # 1. Predict
        predictions = model(batch_X)
        
        # 2. Calculate Error
        loss = criterion(predictions, batch_y)
        
        # 3. Learn (Backpropagation)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/10 | Loss: {loss.item():.4f}")

print("Training Complete!")
```

### Block 8: Evaluating on New Patients
After training, you want to use the model on a new patient's gene expression profile.

```python
# Simulate a NEW patient (never seen by the model)
new_patient_genes = torch.tensor(np.random.rand(1, 50), dtype=torch.float32).to(device)

model.eval() # Set model to evaluation mode (turns off dropout)

# No need to calculate gradients for prediction
with torch.no_grad():
    risk_probability = model(new_patient_genes)
    
print(f"New Patient Disease Risk: {risk_probability.item():.2%}")

if risk_probability > 0.5:
    print("Diagnosis: High Risk of Disease")
else:
    print("Diagnosis: Low Risk (Healthy)")
```

### Summary of Bio-Terms Used

1.  **One-Hot Encoding:** Converting DNA letters (A,C,G,T) into numbers so the math works.
2.  **Gene Expression Matrix:** A table where rows are patients and columns are genes (like Excel).
3.  **1D Convolution (Conv1d):** Sliding a window across a DNA sequence to find patterns.
4.  **Molecular Fingerprint:** A long list of 0s and 1s representing the chemical structure of a drug.
5.  **Overfitting:** When the model memorizes the training data instead of learning general biology (we use Dropout to stop this).
