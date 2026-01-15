import torch
import torch.nn as nn
import torch.optim as optim
import random

# ==============================================================================
# BLOCK 1: SETTING THE STAGE (Imports & Setup)
# ==============================================================================
# We imported:
# - torch: The PyTorch library, the main tool for Deep Learning.
# - torch.nn: Neural Network building blocks (layers, activation functions).
# - torch.optim: Optimizers to help the network learn (adjust weights).
# - random: A standard Python library to generate random numbers.

# Set a "seed" so this code produces the same results every time you run it.
# This is crucial for reproducibility in science.
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)

print("Block 1: Libraries imported and random seed set.")


# ==============================================================================
# BLOCK 2: GENERATING DATA (Bioinformatics Context)
# ==============================================================================
# Goal: Create a dataset of DNA sequences.
# Task: We will generate random DNA sequences.
#       - Half will contain a specific "motif" (pattern) "ACGT". These are "Positive" (label 1).
#       - Half will be random noise. These are "Negative" (label 0).
# In real life, "Positive" might mean "Gene Promoter" or "Protein Binding Site".

def generate_dna_data(num_samples=1000, seq_length=10):
    """
    Generates synthetic DNA data.
    num_samples: Total number of sequences to generate.
    seq_length: Length of each DNA sequence.
    """
    data = []
    labels = []
    
    bases = ['A', 'C', 'G', 'T']
    motif = "ACGT"  # The secret pattern the AI needs to learn
    
    for i in range(num_samples):
        # Python: List comprehension to create a random string of length 10
        # This picks a random base for each position in the sequence length
        sequence = "".join([random.choice(bases) for _ in range(seq_length)])
        
        # We make 50% positive (with motif) and 50% negative (random)
        if i < num_samples // 2:
            # Positive Sample: Insert the motif into the random sequence
            # We overwrite part of the random sequence with "ACGT"
            start_index = random.randint(0, seq_length - len(motif))
            sequence = sequence[:start_index] + motif + sequence[start_index + len(motif):]
            label = 1 # 1 means "Contains Motif"
        else:
            # Negative Sample: Purely random (mostly)
            # (There is a tiny chance it randomly forms ACGT, but we ignore that for now)
            label = 0 # 0 means "Random Noise"
            
        data.append(sequence)
        labels.append(label)
        
    return data, labels

# Generate 2000 examples
raw_sequences, raw_labels = generate_dna_data(num_samples=2000, seq_length=10)

print(f"Block 2: Data generated. Example Positive: {raw_sequences[0]} (Label: {raw_labels[0]})")
print(f"Block 2: Example Negative: {raw_sequences[-1]} (Label: {raw_labels[-1]})")


# ==============================================================================
# BLOCK 3: PREPROCESSING (Encoding DNA for the Machine)
# ==============================================================================
# Computers don't understand 'A' or 'C'. They understand numbers.
# We will use "One-Hot Encoding".
# A -> [1, 0, 0, 0]
# C -> [0, 1, 0, 0]
# G -> [0, 0, 1, 0]
# T -> [0, 0, 0, 1]

def one_hot_encode(sequence):
    """
    Converts a DNA string into a list of numbers.
    Input: "AC"
    Output: [[1,0,0,0], [0,1,0,0]]
    """
    mapping = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    
    encoded_seq = []
    for base in sequence:
        encoded_seq.append(mapping[base])
        
    return encoded_seq

# Test the function
example_seq = "AC"
print(f"Block 3: Encoding '{example_seq}' -> {one_hot_encode(example_seq)}")


# ==============================================================================
# BLOCK 4: THE DATASET PIPELINE (PyTorch Dataset)
# ==============================================================================
# PyTorch uses a 'Dataset' class to manage data and a 'DataLoader' to serve it in batches.
# This prevents loading huge datasets entirely into RAM at once (crucial for big genomic data).

class DNADataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        # This function runs once when we create the object
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        # Tells PyTorch how many items are in the dataset
        return len(self.sequences)

    def __getitem__(self, idx):
        # This is called whenever PyTorch asks for the i-th example
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert to one-hot encoding
        encoded = one_hot_encode(seq)
        
        # Convert Python lists to PyTorch Tensors (the format GPUs love)
        # FloatTensor for inputs (decimal numbers), LongTensor for labels (integers)
        # We flatten the input: A 10-base sequence becomes a vector of size 40 (10x4) 
        x = torch.tensor(encoded, dtype=torch.float32).flatten() 
        y = torch.tensor(label, dtype=torch.float32) # Using float for BCELoss later
        
        return x, y

# Initialize Dataset
dataset = DNADataset(raw_sequences, raw_labels)

# Split into Train (80%) and Test (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

# Create DataLoaders
# batch_size=32 means the model will learn from 32 examples at a time
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Block 4: Dataset and DataLoaders created.")


# ==============================================================================
# BLOCK 5: THE MODEL (The "Brain")
# ==============================================================================
# We build a simple Multi-Layer Perceptron (MLP).
# Structure: Input -> Hidden Layer -> Output
# Input size: 40 (length 10 * 4 bases)
# Output size: 1 (Probability of being "Positive")

class SimpleDNANet(nn.Module):
    def __init__(self):
        super(SimpleDNANet, self).__init__()
        
        # Layer 1: Linear transformation. 
        # Takes 40 inputs, creates 16 hidden features.
        self.layer1 = nn.Linear(in_features=40, out_features=16)
        
        # Activation Function: ReLU (Rectified Linear Unit)
        # Turns negative numbers to 0. Adds non-linearity so we can learn complex patterns.
        self.relu = nn.ReLU()
        
        # Layer 2: Output layer.
        # Takes 16 hidden features, outputs 1 single number (score).
        self.layer2 = nn.Linear(in_features=16, out_features=1)
        
        # Activation Function: Sigmoid
        # Squishes the output score between 0 and 1 (like a probability).
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # This defines how data flows through the network
        x = self.layer1(x)  # Step 1: Input to Hidden
        x = self.relu(x)    # Step 2: Apply activation
        x = self.layer2(x)  # Step 3: Hidden to Output
        x = self.sigmoid(x) # Step 4: Convert to probability
        return x

model = SimpleDNANet()
print("Block 5: Model architecture defined.")
print(model)


# ==============================================================================
# BLOCK 6: TRAINING (The Learning Process)
# ==============================================================================
# 1. Loss Function: Measures how "wrong" the model is.
#    Binary Cross Entropy (BCELoss) is standard for Yes/No classification.
criterion = nn.BCELoss()

# 2. Optimizer: Adjusts the model's numbers (weights) to reduce the loss.
#    SGD (Stochastic Gradient Descent) is the classic algorithm.
#    lr (Learning Rate) controls how big the steps are.
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Block 6: Starting training...")

epochs = 20 # Number of times we loop through the entire dataset

for epoch in range(epochs):
    total_loss = 0
    
    # Loop over batches of data
    for batch_features, batch_labels in train_loader:
        # A. RESET GRADIENTS
        # Clear old calculations from previous step
        optimizer.zero_grad()
        
        # B. FORWARD PASS
        # Ask the model for predictions
        outputs = model(batch_features)
        
        # C. COMPUTE LOSS
        # Compare predictions (outputs) with reality (batch_labels)
        # batch_labels needs to be reshaped to match outputs dimensions
        loss = criterion(outputs, batch_labels.view(-1, 1))
        
        # D. BACKWARD PASS (Backpropagation)
        # Calculate how much each weight contributed to the error
        loss.backward()
        
        # E. OPTIMIZE
        # Update weights to fix the error
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print average loss every 5 epochs
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {total_loss / len(train_loader):.4f}")


# ==============================================================================
# BLOCK 7: EVALUATION (Did it learn?)
# ==============================================================================
# We test the model on data it has NEVER seen before (test_dataset).

print("\nBlock 7: Evaluating on Test Data...")
model.eval() # Switch to evaluation mode (turns off training-specific features)

correct = 0
total = 0

with torch.no_grad(): # Don't calculate gradients during testing (saves memory)
    for inputs, labels in test_loader:
        outputs = model(inputs)
        
        # If output > 0.5, predict 1 (Positive), else 0 (Negative)
        predicted = (outputs > 0.5).float()
        
        total += labels.size(0)
        correct += (predicted.view(-1) == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on Test Set: {accuracy:.2f}%")

# Let's verify manually with our examples
print("\n--- Manual Check ---")
def predict_dna(dna_string):
    encoded = one_hot_encode(dna_string)
    tensor_input = torch.tensor(encoded, dtype=torch.float32).flatten().unsqueeze(0) # Add batch dimension
    prediction = model(tensor_input).item()
    return prediction

# Test with a made-up positive sequence (contains ACGT)
test_pos = "TTTACGTTTT" 
prob_pos = predict_dna(test_pos)
print(f"Seq: {test_pos} (Has ACGT) -> Model says: {prob_pos:.4f} ({'Positive' if prob_pos > 0.5 else 'Negative'})")

# Test with a made-up negative sequence
test_neg = "TTTTTTTTTT"
prob_neg = predict_dna(test_neg)
print(f"Seq: {test_neg} (Random)   -> Model says: {prob_neg:.4f} ({'Positive' if prob_neg > 0.5 else 'Negative'})")
