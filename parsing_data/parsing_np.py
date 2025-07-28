import gzip
import numpy as np
from collections import defaultdict
import multiprocessing as mp
from functools import partial
import os
from typing import List, Tuple, Iterator, Optional
import time

class FastqProcessor:
    """
    Fast FASTQ processor using numpy for quality operations and efficient clustering.
    """
    
    def __init__(self, min_length: int = 100, min_quality: int = 20, quality_window: int = 4):
        self.min_length = min_length
        self.min_quality = min_quality
        self.quality_window = quality_window
        
    def parse_fastq(self, filename: str) -> Iterator[Tuple[str, str, str]]:
        """
        Memory-efficient FASTQ parser that yields (header, sequence, quality) tuples.
        Handles both .fastq and .fastq.gz files.
        """
        open_func = gzip.open if filename.endswith('.gz') else open
        
        with open_func(filename, 'rt') as f:
            while True:
                header = f.readline().strip()
                if not header:
                    break
                
                sequence = f.readline().strip()
                plus = f.readline().strip()
                quality = f.readline().strip()
                
                if header and sequence and quality:
                    yield header[1:], sequence, quality  # Remove '@' from header
                    
    def quality_scores_to_numpy(self, quality_string: str, offset: int = 33) -> np.ndarray:
        """
        Convert quality string to numpy array of quality scores.
        Default offset=33 for Sanger/Illumina 1.8+ format.
        """
        return np.array([ord(c) - offset for c in quality_string], dtype=np.uint8)
    
    def sliding_window_quality(self, quality_scores: np.ndarray) -> np.ndarray:
        """
        Calculate sliding window average quality using numpy for speed.
        """
        if len(quality_scores) < self.quality_window:
            return quality_scores
            
        # Use numpy convolution for fast sliding window
        kernel = np.ones(self.quality_window) / self.quality_window
        return np.convolve(quality_scores, kernel, mode='valid')
    
    def trim_sequence(self, sequence: str, quality: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Trim sequence based on quality scores using numpy operations.
        Returns None, None if sequence doesn't meet criteria.
        """
        quality_scores = self.quality_scores_to_numpy(quality)
        
        # Find first position where sliding window quality drops below threshold
        if len(quality_scores) >= self.quality_window:
            window_qualities = self.sliding_window_quality(quality_scores)
            
            # Find first position where quality drops
            low_quality_positions = np.where(window_qualities < self.min_quality)[0]
            
            if len(low_quality_positions) > 0:
                trim_pos = low_quality_positions[0]
            else:
                trim_pos = len(sequence)
        else:
            # For short sequences, use mean quality
            if np.mean(quality_scores) < self.min_quality:
                return None, None
            trim_pos = len(sequence)
        
        # Trim sequence and quality
        trimmed_seq = sequence[:trim_pos]
        trimmed_qual = quality[:trim_pos]
        
        # Check minimum length
        if len(trimmed_seq) < self.min_length:
            return None, None
            
        return trimmed_seq, trimmed_qual
    
    def process_chunk(self, sequences_chunk: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """
        Process a chunk of sequences for multiprocessing.
        """
        results = []
        for header, sequence, quality in sequences_chunk:
            trimmed_seq, trimmed_qual = self.trim_sequence(sequence, quality)
            if trimmed_seq is not None:
                results.append((header, trimmed_seq, trimmed_qual))
        return results
    
    def process_fastq_file(self, filename: str, chunk_size: int = 10000, 
                          n_processes: Optional[int] = None) -> List[Tuple[str, str, str]]:
        """
        Process entire FASTQ file with multiprocessing for speed.
        """
        if n_processes is None:
            n_processes = mp.cpu_count()
        
        print(f"Processing {filename} with {n_processes} processes...")
        
        # Read sequences in chunks
        chunks = []
        current_chunk = []
        
        for i, (header, sequence, quality) in enumerate(self.parse_fastq(filename)):
            current_chunk.append((header, sequence, quality))
            
            if len(current_chunk) >= chunk_size:
                chunks.append(current_chunk)
                current_chunk = []
                
            if i % 50000 == 0 and i > 0:
                print(f"Loaded {i} sequences...")
        
        if current_chunk:
            chunks.append(current_chunk)
        
        print(f"Processing {len(chunks)} chunks...")
        
        # Process chunks in parallel
        with mp.Pool(n_processes) as pool:
            results = pool.map(self.process_chunk, chunks)
        
        # Flatten results
        all_sequences = []
        for chunk_result in results:
            all_sequences.extend(chunk_result)
        
        print(f"Kept {len(all_sequences)} sequences after quality trimming")
        return all_sequences

class SequenceClusterer:
    """
    Efficient sequence clustering for 97% similarity using k-mer based pre-filtering.
    """
    
    def __init__(self, similarity_threshold: float = 0.97, kmer_size: int = 8):
        self.similarity_threshold = similarity_threshold
        self.kmer_size = kmer_size
    
    def get_kmers(self, sequence: str) -> set:
        """Extract k-mers from sequence."""
        return {sequence[i:i+self.kmer_size] for i in range(len(sequence)-self.kmer_size+1)}
    
    def jaccard_similarity(self, kmers1: set, kmers2: set) -> float:
        """Calculate Jaccard similarity between two k-mer sets."""
        intersection = len(kmers1 & kmers2)
        union = len(kmers1 | kmers2)
        return intersection / union if union > 0 else 0.0
    
    def hamming_distance_numpy(self, seq1: str, seq2: str) -> float:
        """
        Fast Hamming distance calculation using numpy.
        Only works for sequences of equal length.
        """
        if len(seq1) != len(seq2):
            return 0.0  # Different lengths = not similar enough
        
        # Convert to numpy arrays for fast comparison
        arr1 = np.array(list(seq1))
        arr2 = np.array(list(seq2))
        
        # Calculate similarity
        matches = np.sum(arr1 == arr2)
        return matches / len(seq1)
    
    def sequence_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate sequence similarity. Uses different methods based on length difference.
        """
        len_diff = abs(len(seq1) - len(seq2))
        max_len = max(len(seq1), len(seq2))
        
        # If length difference is too large, sequences can't be 97% similar
        if len_diff / max_len > (1 - self.similarity_threshold):
            return 0.0
        
        # For similar lengths, use Hamming distance on the shorter length
        if len_diff / max_len < 0.1:  # Less than 10% length difference
            min_len = min(len(seq1), len(seq2))
            return self.hamming_distance_numpy(seq1[:min_len], seq2[:min_len])
        
        # For different lengths, use k-mer based similarity
        kmers1 = self.get_kmers(seq1)
        kmers2 = self.get_kmers(seq2)
        return self.jaccard_similarity(kmers1, kmers2)
    
    def cluster_sequences(self, sequences: List[Tuple[str, str, str]], 
                         max_comparisons: int = 1000000) -> List[List[int]]:
        """
        Cluster sequences based on similarity threshold.
        Returns list of clusters, where each cluster is a list of sequence indices.
        """
        print(f"Clustering {len(sequences)} sequences...")
        
        clusters = []
        clustered = set()
        
        # Pre-compute k-mers for k-mer based filtering
        sequence_kmers = []
        for _, seq, _ in sequences:
            sequence_kmers.append(self.get_kmers(seq))
        
        comparisons_made = 0
        
        for i, (header1, seq1, qual1) in enumerate(sequences):
            if i in clustered:
                continue
                
            if i % 1000 == 0:
                print(f"Processing sequence {i}/{len(sequences)}")
            
            current_cluster = [i]
            clustered.add(i)
            
            # Compare with remaining sequences
            for j in range(i + 1, len(sequences)):
                if j in clustered or comparisons_made >= max_comparisons:
                    continue
                
                # Quick k-mer pre-filter
                kmer_sim = self.jaccard_similarity(sequence_kmers[i], sequence_kmers[j])
                if kmer_sim < 0.5:  # Rough pre-filter
                    continue
                
                header2, seq2, qual2 = sequences[j]
                similarity = self.sequence_similarity(seq1, seq2)
                comparisons_made += 1
                
                if similarity >= self.similarity_threshold:
                    current_cluster.append(j)
                    clustered.add(j)
            
            if len(current_cluster) > 0:
                clusters.append(current_cluster)
        
        print(f"Created {len(clusters)} clusters from {len(sequences)} sequences")
        print(f"Made {comparisons_made} sequence comparisons")
        
        return clusters

# Example usage and workflow
def main():
    # Initialize processor
    processor = FastqProcessor(min_length=100, min_quality=20)
    
    # Process FASTQ file
    filename = "example.fastq.gz"  # Replace with your file
    
    if os.path.exists(filename):
        # Process file
        start_time = time.time()
        processed_sequences = processor.process_fastq_file(filename)
        processing_time = time.time() - start_time
        
        print(f"Processing took {processing_time:.2f} seconds")
        
        # Convert to numpy arrays for further analysis
        sequences_only = [seq for _, seq, _ in processed_sequences]
        qualities_only = [qual for _, _, qual in processed_sequences]
        
        # Example: Calculate sequence length distribution using numpy
        seq_lengths = np.array([len(seq) for seq in sequences_only])
        print(f"Sequence length stats:")
        print(f"  Mean: {np.mean(seq_lengths):.1f}")
        print(f"  Median: {np.median(seq_lengths):.1f}")
        print(f"  Std: {np.std(seq_lengths):.1f}")
        
        # Cluster sequences (use subset for demo)
        if len(processed_sequences) > 1000:
            print("Using first 1000 sequences for clustering demo...")
            subset = processed_sequences[:1000]
        else:
            subset = processed_sequences
        
        clusterer = SequenceClusterer(similarity_threshold=0.97)
        clusters = clusterer.cluster_sequences(subset)
        
        # Analyze clusters
        cluster_sizes = [len(cluster) for cluster in clusters]
        print(f"\nCluster analysis:")
        print(f"  Total clusters: {len(clusters)}")
        print(f"  Singleton clusters: {sum(1 for size in cluster_sizes if size == 1)}")
        print(f"  Largest cluster size: {max(cluster_sizes) if cluster_sizes else 0}")

if __name__ == "__main__":
    main()
