import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import hdbscan
import umap
import gzip
import os

def load_sequences(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    if 'fastq' in file_path.lower():
        file_format = 'fastq'
    elif 'fasta' in file_path.lower() or 'fa' in file_path.lower():
        file_format = 'fasta'
    else:
        file_format = 'fasta'

    print(f"Processing {file_path} as {file_format}...")

    sequences = []
    ids = []
    
    open_func = gzip.open if file_path.endswith('.gz') else open
    
    with open_func(file_path, 'rt') as handle:
        for record in SeqIO.parse(handle, file_format):
            sequences.append(str(record.seq).upper())
            ids.append(record.id)
            
    return ids, sequences

def get_kmer_vectors(sequences, k=4):
    def kmer_tokenizer(seq):
        return [seq[i:i+k] for i in range(len(seq) - k + 1)]
    
    vectorizer = CountVectorizer(tokenizer=kmer_tokenizer, lowercase=False)
    X = vectorizer.fit_transform(sequences)
    return X.toarray()

input_file = "my_sequences.fastq.gz" 

try:
    ids, sequences = load_sequences(input_file)
except FileNotFoundError as e:
    print(e)
    exit()

X = get_kmer_vectors(sequences, k=4)

reducer = umap.UMAP(n_neighbors=15, n_components=2, metric='manhattan', random_state=42)
embedding = reducer.fit_transform(X)

clusterer = hdbscan.HDBSCAN(min_cluster_size=10, gen_min_span_tree=True)
cluster_labels = clusterer.fit_predict(embedding)

df = pd.DataFrame(embedding, columns=['UMAP_1', 'UMAP_2'])
df['Cluster'] = cluster_labels
df['Cluster'] = df['Cluster'].astype(str)
df = df.sort_values('Cluster') 

plt.figure(figsize=(12, 10))
sns.scatterplot(
    data=df, 
    x='UMAP_1', 
    y='UMAP_2', 
    hue='Cluster', 
    palette='bright',
    s=40,
    alpha=0.7,
    edgecolor=None
)

plt.title(f'Cluster Analysis of {input_file}')
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
