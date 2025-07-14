#!/usr/bin/env python3

import numpy as np
import cupy as cp
import time

# --- Re-use code from the previous step ---

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
        if (prev_base >= 4 || new_base >= 4) {
            valid_kmer = false; continue;
        }
        if (!valid_kmer) {
            kmer_value = 0;
            bool can_recalculate = true;
            for (int j = 0; j < k; ++j) {
                unsigned char current_base = sequences[seq_id * seq_len + i + j];
                if (current_base >= 4) { can_recalculate = false; break; }
                kmer_value = kmer_value * 4 + current_base;
            }
            if (can_recalculate) {
                valid_kmer = true;
                atomicAdd(&kmer_counts[kmer_value], 1);
            }
        } else {
            kmer_value = (kmer_value - prev_base * first_base_multiplier) * 4 + new_base;
            atomicAdd(&kmer_counts[kmer_value], 1);
        }
    }
}
''', 'kmer_counter')

# --- New Machine Learning Code ---

def create_synthetic_data(base_counts, num_samples=1000):
    """Creates a synthetic dataset for classification."""
    # Normalize the base counts to get a probability distribution
    base_profile = base_counts / base_counts.sum()

    # Create a second, "diseased" profile by slightly altering the first one
    diseased_profile = base_profile.copy()
    # Artificially increase the count of some k-mers and decrease others
    indices_to_increase = np.random.choice(len(diseased_profile), 50, replace=False)
    indices_to_decrease = np.random.choice(len(diseased_profile), 50, replace=False)
    diseased_profile[indices_to_increase] *= 1.5
    diseased_profile[indices_to_decrease] *= 0.5
    diseased_profile /= diseased_profile.sum() # Re-normalize

    # Generate synthetic samples
    # Each sample is a k-mer count vector drawn from a multinomial distribution
    healthy_samples = np.random.multinomial(10000, base_profile, size=num_samples // 2)
    diseased_samples = np.random.multinomial(10000, diseased_profile, size=num_samples // 2)

    # Combine, create labels, and shuffle
    X = np.vstack([healthy_samples, diseased_samples]).astype(np.float32)
    y = np.hstack([np.zeros(num_samples // 2), np.ones(num_samples // 2)]).astype(np.float32)
    
    shuffle_idx = np.random.permutation(num_samples)
    return X[shuffle_idx], y[shuffle_idx]

def sigmoid(x):
    """Sigmoid activation function for logistic regression."""
    return 1 / (1 + cp.exp(-x))

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
    cp.cuda.Stream.null.synchronize()
    kmer_counts_cpu = cp.asnumpy(kmer_counts_gpu)
    print(f"   Base profile generated. Total k-mers found: {kmer_counts_cpu.sum():,}")

    # --- 2. Create Synthetic Dataset ---
    print("\n2. Creating synthetic training data...")
    X_cpu, y_cpu = create_synthetic_data(kmer_counts_cpu, num_samples=2000)
    print(f"   Dataset created with {X_cpu.shape[0]} samples.")

    # --- 3. Train Logistic Regression on GPU ---
    print("\n3. Training Logistic Regression model on GPU...")
    # Move training data to the GPU
    X_gpu = cp.asarray(X_cpu)
    y_gpu = cp.asarray(y_cpu).reshape(-1, 1) # Reshape for matrix operations

    # Initialize model parameters on the GPU
    num_features = X_gpu.shape[1]
    weights = cp.zeros((num_features, 1), dtype=cp.float32)
    bias = cp.zeros(1, dtype=cp.float32)

    # Hyperparameters
    learning_rate = 0.01
    epochs = 100

    start_time = time.time()
    for epoch in range(epochs):
        # --- All of these are matrix/vector operations happening on the GPU ---
        
        # Forward pass: Make predictions
        linear_model = cp.dot(X_gpu, weights) + bias
        y_predicted = sigmoid(linear_model)

        # Calculate the loss (Binary Cross-Entropy)
        loss = -cp.mean(y_gpu * cp.log(y_predicted) + (1 - y_gpu) * cp.log(1 - y_predicted))

        # Backward pass: Calculate gradients
        dw = (1 / num_features) * cp.dot(X_gpu.T, (y_predicted - y_gpu))
        db = (1 / num_features) * cp.sum(y_predicted - y_gpu)

        # Update weights and bias
        weights -= learning_rate * dw
        bias -= learning_rate * db

        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:3}/{epochs} | Loss: {loss:.4f}")
    
    cp.cuda.Stream.null.synchronize() # Wait for all GPU operations to finish
    end_time = time.time()
    gpu_time = end_time - start_time

    print(f"\nGPU training finished in {gpu_time:.4f} seconds.")

    # --- 4. Evaluate the Model ---
    # Make final predictions on the training data
    linear_model = cp.dot(X_gpu, weights) + bias
    y_predicted = sigmoid(linear_model)
    predictions_binary = (y_predicted > 0.5).astype(cp.int32)
    
    accuracy = cp.mean((predictions_binary == y_gpu).astype(cp.float32))

    print(f"Final training accuracy: {accuracy * 100:.2f}%")

