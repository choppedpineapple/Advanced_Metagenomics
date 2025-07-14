

import numpy as np
import cupy as cp
import time
import torch
import torch.nn as nn

# --- Part 1: Feature Extraction (Identical to the previous script) ---
# We still use our efficient CuPy kernel for the custom task of k-mer counting.

FASTQ_PATH = "/home/abhi/workspace/1.datasets/16S_gut_microbiome/sample_1/SRR32461054_1.fastq"
K = 8
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

def parse_fastq_sequences(file_path):
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                yield line.strip()

def sequences_to_numerical(sequences):
    max_len = max(len(s) for s in sequences)
    arr = np.full((len(sequences), max_len), fill_value=BASE_TO_INT['N'], dtype=np.uint8)
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            arr[i, j] = BASE_TO_INT.get(base, BASE_TO_INT['N'])
    return arr

kmer_counter_kernel = cp.RawKernel(r'''
extern "C" __global__
void kmer_counter(const unsigned char* sequences, unsigned long long* kmer_counts, int num_sequences, int seq_len, int k) {
    int seq_id = blockDim.x * blockIdx.x + threadIdx.x;
    if (seq_id >= num_sequences) return;

    unsigned long long kmer_value = 0;
    unsigned long long first_base_multiplier = 1;
    for (int i = 0; i < k - 1; ++i) first_base_multiplier *= 4;

    bool valid_kmer = true;
    for (int i = 0; i < k; ++i) {
        unsigned char base = sequences[seq_id * seq_len + i];
        if (base >= 4) { valid_kmer = false; break; }
        kmer_value = kmer_value * 4 + base;
    }
    if (valid_kmer) atomicAdd(&kmer_counts[kmer_value], 1);

    for (int i = 1; i <= seq_len - k; ++i) {
        unsigned char prev_base = sequences[seq_id * seq_len + i - 1];
        unsigned char new_base = sequences[seq_id * seq_len + i + k - 1];
        if (prev_base >= 4 || new_base >= 4) { valid_kmer = false; continue; }
        if (!valid_kmer) {
            kmer_value = 0;
            bool can_recalculate = true;
            for (int j = 0; j < k; ++j) {
                unsigned char current_base = sequences[seq_id * seq_len + i + j];
                if (current_base >= 4) { can_recalculate = false; break; }
                kmer_value = kmer_value * 4 + current_base;
            }
            if (can_recalculate) { valid_kmer = true; atomicAdd(&kmer_counts[kmer_value], 1); }
        } else {
            kmer_value = (kmer_value - prev_base * first_base_multiplier) * 4 + new_base;
            atomicAdd(&kmer_counts[kmer_value], 1);
        }
    }
}
''', 'kmer_counter')

def create_synthetic_data(base_counts, num_samples=1000):
    base_profile = base_counts / base_counts.sum()
    diseased_profile = base_profile.copy()
    indices_to_increase = np.random.choice(len(diseased_profile), 50, replace=False)
    indices_to_decrease = np.random.choice(len(diseased_profile), 50, replace=False)
    diseased_profile[indices_to_increase] *= 1.5
    diseased_profile[indices_to_decrease] *= 0.5
    diseased_profile /= diseased_profile.sum()
    healthy_samples = np.random.multinomial(10000, base_profile, size=num_samples // 2)
    diseased_samples = np.random.multinomial(10000, diseased_profile, size=num_samples // 2)
    X = np.vstack([healthy_samples, diseased_samples]).astype(np.float32)
    y = np.hstack([np.zeros(num_samples // 2), np.ones(num_samples // 2)]).astype(np.float32)
    shuffle_idx = np.random.permutation(num_samples)
    return X[shuffle_idx], y[shuffle_idx]

# --- Part 2: PyTorch Model Definition ---

class PyTorchLogisticRegression(nn.Module):
    def __init__(self, num_features):
        super(PyTorchLogisticRegression, self).__init__()
        # Define a single linear layer. This one line replaces our manual
        # creation of 'weights' and 'bias' tensors.
        self.linear = nn.Linear(num_features, 1)

    def forward(self, x):
        # Defines how to get from an input x to an output.
        # PyTorch automatically applies the sigmoid function.
        return torch.sigmoid(self.linear(x))

if __name__ == "__main__":
    # --- 1. Generate Base Profile (Same as before) ---
    print("1. Generating base k-mer profile from FASTQ...")
    sequences_cpu = list(parse_fastq_sequences(FASTQ_PATH))
    numerical_seqs_cpu = sequences_to_numerical(sequences_cpu)
    num_sequences, seq_len = numerical_seqs_cpu.shape
    sequences_gpu = cp.asarray(numerical_seqs_cpu)
    kmer_counts_gpu = cp.zeros(4**K, dtype=cp.uint64)
    threads_per_block = 256
    grid_size = (num_sequences + threads_per_block - 1) // threads_per_block
    kmer_counter_kernel((grid_size,), (threads_per_block,), (sequences_gpu, kmer_counts_gpu, num_sequences, seq_len, K))
    kmer_counts_cpu = cp.asnumpy(kmer_counts_gpu)
    print(f"   Base profile generated.")

    # --- 2. Create Synthetic Dataset ---
    print("\n2. Creating synthetic training data...")
    X_cpu, y_cpu = create_synthetic_data(kmer_counts_cpu, num_samples=2000)
    print(f"   Dataset created with {X_cpu.shape[0]} samples.")

    # --- 3. Train with PyTorch on GPU ---
    print("\n3. Training Logistic Regression model with PyTorch on GPU...")
    
    # Set the device to 'cuda' (the GPU) if available, otherwise CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"   Using device: {device}")

    # Convert NumPy arrays to PyTorch Tensors and move them to the GPU
    X_train = torch.from_numpy(X_cpu).to(device)
    y_train = torch.from_numpy(y_cpu).reshape(-1, 1).to(device)

    # Initialize the model, loss function, and optimizer
    num_features = X_train.shape[1]
    model = PyTorchLogisticRegression(num_features).to(device) # Move model to GPU
    
    # These two lines replace all our manual gradient and loss math!
    loss_function = nn.BCELoss() # Binary Cross-Entropy Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # The PyTorch Training Loop
    epochs = 100
    start_time = time.time()
    for epoch in range(epochs):
        # Forward pass: model makes a prediction
        y_pred = model(X_train)

        # Calculate the loss
        loss = loss_function(y_pred, y_train)

        # Backward pass: PyTorch calculates all the gradients automatically!
        loss.backward()

        # Optimizer step: PyTorch updates all the model's weights
        optimizer.step()

        # Zero the gradients to prevent them from accumulating
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3}/{epochs} | Loss: {loss.item():.4f}")
    
    end_time = time.time()
    pytorch_time = end_time - start_time
    print(f"\nPyTorch GPU training finished in {pytorch_time:.4f} seconds.")

    # --- 4. Evaluate the Model ---
    # No need for a special mode, just make predictions
    y_pred_final = model(X_train)
    # Convert probabilities to binary 0 or 1 predictions
    preds = (y_pred_final > 0.5).float()
    
    accuracy = (preds == y_train).float().mean()
    print(f"Final training accuracy: {accuracy.item() * 100:.2f}%")

