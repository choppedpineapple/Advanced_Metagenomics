import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd

# Example: DNA sequences
sequences = [
    "ATCGATCG", "ATCGATCC", "ATCGATCA",  # Similar group
    "GCTAGCTA", "GCTAGCTT", "GCTAGCTG",  # Another group
    "TTTTAAAA", "AAAATTTT",               # Outliers/noise
]

# Step 1: Encode DNA (k-mer frequency encoding)
def kmer_features(seqs, k=3):
    """Convert sequences to k-mer frequency vectors"""
    from itertools import product
    kmers = [''.join(p) for p in product('ACGT', repeat=k)]
    features = []
    for seq in seqs:
        seq = seq.upper()
        counts = {kmer: 0 for kmer in kmers}
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            if kmer in counts:
                counts[kmer] += 1
        features.append([counts[k] for k in kmers])
    return np.array(features)

X = kmer_features(sequences, k=2)  # 16-dim vectors for dimers

# Step 2: Reduce to 3D for visualization
pca = PCA(n_components=3)
X_3d = pca.fit_transform(X)

# Alternative: t-SNE for better separation (slower)
# from sklearn.manifold import TSNE
# X_3d = TSNE(n_components=3, perplexity=2).fit_transform(X)

# Step 3: DBSCAN clustering
dbscan = DBSCAN(eps=0.5, min_samples=2)
labels = dbscan.fit_predict(X_3d)  # -1 indicates noise points

# Step 4: 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Color map: -1 (noise) = black, others = distinct colors
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
for label in set(labels):
    mask = labels == label
    color = 'black' if label == -1 else colors[label]
    label_name = f'Noise' if label == -1 else f'Cluster {label}'
    
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2], 
               c=[color], label=label_name, s=100, alpha=0.8, edgecolors='k')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%})')
ax.set_title('DBSCAN Clustering of DNA Sequences (3D PCA Projection)')
ax.legend()

plt.tight_layout()
plt.show()

# Print results
df = pd.DataFrame({
    'Sequence': sequences,
    'Cluster': labels,
    'PC1': X_3d[:, 0],
    'PC2': X_3d[:, 1],
    'PC3': X_3d[:, 2]
})
print(df)
