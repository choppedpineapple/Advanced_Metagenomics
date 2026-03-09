import pandas as pd
from Bio import SeqIO
from sklearn.feature_extraction.text import CountVectorizer
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def process_scfv_data(fasta_file):
    print("1. Reading that massive FASTA file...")
    # Read the fasta and count exact sequence frequencies
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    
    # Dereplicate
    seq_counts = pd.Series(sequences).value_counts().reset_index()
    seq_counts.columns = ['Sequence', 'Count']
    
    # Expecting 100-300 unique clones. Taking the top 300.
    top_clones = seq_counts.head(300).copy()
    print(f"Found {len(seq_counts)} total unique reads. Taking the top {len(top_clones)} for UMAP.")

    print("2. Chopping sequences into k-mers...")
    # Using 5-mers (chunks of 5 nucleotides). 
    # CountVectorizer turns DNA into a matrix of k-mer frequencies.
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
    kmer_matrix = vectorizer.fit_transform(top_clones['Sequence']).toarray()

    print("3. Doing the UMAP magic...")
    # Initialize UMAP. Using cosine metric because it works well for frequency data.
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(kmer_matrix)

    # Add the 2D coordinates back to our dataframe
    top_clones['UMAP_1'] = embedding[:, 0]
    top_clones['UMAP_2'] = embedding[:, 1]

    print("4. Painting the masterpiece...")
    plt.figure(figsize=(10, 8))
    # Size the dots by how frequent the clone is (Count)
    scatter = sns.scatterplot(
        data=top_clones, 
        x='UMAP_1', y='UMAP_2', 
        size='Count', sizes=(20, 500), 
        alpha=0.7, color='purple'
    )
    plt.title("SeqUMAP of scFv Clones")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return top_clones

# Run
# df = process_scfv_data("merged.fasta")
