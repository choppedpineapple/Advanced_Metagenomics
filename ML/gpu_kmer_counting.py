import numpy as np
import cupy as cp
import time

FASTQ_PATH = "/home/abhi/workspace/1.datasets/16S_gut_microbiome/sample_1/SRR32461054_1.fastq"
K = 8 # We will count 8-mers

# Mapping from DNA base to an integer
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}

# --- 1. Helper Functions (CPU) ---

def parse_fastq_sequences(file_path):
    """Parses a FASTQ file and yields the sequence for each read."""
    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                yield line.strip()

def sequences_to_numerical(sequences):
    """Converts a list of DNA sequences to a NumPy array of integers."""
    # Find the length of the longest sequence to make all arrays the same size
    max_len = max(len(s) for s in sequences)
    # Create a NumPy array filled with a padding value (e.g., 4 for 'N')
    # Using 'uint8' is memory-efficient as we only have 5 possible values
    arr = np.full((len(sequences), max_len), fill_value=BASE_TO_INT['N'], dtype=np.uint8)
    # Fill the array with the numerical representation of each sequence
    for i, seq in enumerate(sequences):
        for j, base in enumerate(seq):
            arr[i, j] = BASE_TO_INT.get(base, BASE_TO_INT['N']) # Default to N if base is unknown
    return arr

# --- 2. Custom CUDA Kernel ---

# This C++ code will be compiled by CuPy and run on the GPU.
# It's designed to count k-mers for many sequences in parallel.
kmer_counter_kernel = cp.RawKernel(r'''
extern "C" __global__
void kmer_counter(const unsigned char* sequences, unsigned long long* kmer_counts, int num_sequences, int seq_len, int k) {
    int seq_id = blockDim.x * blockIdx.x + threadIdx.x;

    if (seq_id >= num_sequences) {
        return;
    }

    unsigned long long kmer_value = 0;
    unsigned long long first_base_multiplier = 1;
    for (int i = 0; i < k - 1; ++i) {
        first_base_multiplier *= 4;
    }

    // Calculate the first k-mer
    bool valid_kmer = true;
    for (int i = 0; i < k; ++i) {
        unsigned char base = sequences[seq_id * seq_len + i];
        if (base >= 4) { // Handle 'N' or padding
            valid_kmer = false;
            break;
        }
        kmer_value = kmer_value * 4 + base;
    }

    if (valid_kmer) {
        atomicAdd(&kmer_counts[kmer_value], 1);
    }

    // Slide the window across the rest of the sequence
    for (int i = 1; i <= seq_len - k; ++i) {
        unsigned char prev_base = sequences[seq_id * seq_len + i - 1];
        unsigned char new_base = sequences[seq_id * seq_len + i + k - 1];

        if (prev_base >= 4 || new_base >= 4) {
            valid_kmer = false;
            continue; // Skip this window if it contains 'N'
        }

        // If the previous k-mer was invalid, we need to recalculate from scratch
        if (!valid_kmer) {
            kmer_value = 0;
            bool can_recalculate = true;
            for (int j = 0; j < k; ++j) {
                unsigned char current_base = sequences[seq_id * seq_len + i + j];
                if (current_base >= 4) {
                    can_recalculate = false;
                    break;
                }
                kmer_value = kmer_value * 4 + current_base;
            }
            if (can_recalculate) {
                valid_kmer = true;
                atomicAdd(&kmer_counts[kmer_value], 1);
            }
        } else {
            // Efficiently update the k-mer value using the sliding window
            kmer_value = (kmer_value - prev_base * first_base_multiplier) * 4 + new_base;
            atomicAdd(&kmer_counts[kmer_value], 1);
        }
    }
}
''', 'kmer_counter')


if __name__ == "__main__":
    # --- CPU SETUP ---
    print("1. Parsing and converting sequences on CPU...")
    sequences_cpu = list(parse_fastq_sequences(FASTQ_PATH))
    numerical_seqs_cpu = sequences_to_numerical(sequences_cpu)
    num_sequences, seq_len = numerical_seqs_cpu.shape
    print(f"   Found {num_sequences} sequences, max length {seq_len}.")

    # --- GPU EXECUTION ---
    print("\n2. Running k-mer counting on GPU...")
    # Move data to the GPU
    sequences_gpu = cp.asarray(numerical_seqs_cpu)
    
    # Create an array on the GPU to store the counts. The size is 4^k.
    kmer_counts_gpu = cp.zeros(4**K, dtype=cp.uint64)

    # Define grid and block dimensions for the kernel launch
    threads_per_block = 256
    grid_size = (num_sequences + threads_per_block - 1) // threads_per_block

    start_time = time.time()
    # Launch the kernel!
    kmer_counter_kernel(
        (grid_size,), 
        (threads_per_block,), 
        (sequences_gpu, kmer_counts_gpu, num_sequences, seq_len, K)
    )
    cp.cuda.Stream.null.synchronize() # Wait for GPU to finish
    end_time = time.time()
    gpu_time = end_time - start_time
    print(f"   GPU execution took: {gpu_time:.4f} seconds.")

    # Copy results back to CPU
    kmer_counts_result = cp.asnumpy(kmer_counts_gpu)

    # --- Verification & Output ---
    print("\n3. Top 10 most frequent k-mers found by GPU:")
    # Get the indices of the top 10 k-mers
    top_10_indices = np.argsort(kmer_counts_result)[-10:][::-1]

    # Function to convert a numerical k-mer index back to a DNA string
    int_to_base = ['A', 'C', 'G', 'T']
    def index_to_kmer(index, k):
        kmer = ""
        for _ in range(k):
            index, base_val = divmod(index, 4)
            kmer = int_to_base[base_val] + kmer
        return kmer

    for index in top_10_indices:
        kmer_str = index_to_kmer(index, K)
        count = kmer_counts_result[index]
        print(f"   K-mer: {kmer_str}  |  Count: {count:,}")
