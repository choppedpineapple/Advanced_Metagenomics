import os
import pandas as pd
from Bio import SeqIO
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading and Label Parsing ---
def parse_label(header, delimiter=';', level=-1):
    return header.split(delimiter)[level]

def load_data(file_path):
    data = []
    for record in SeqIO.parse(file_path, "fasta"):
        header = record.description
        sequence = str(record.seq).upper()  # Ensure uniform case
        label = parse_label(header)
        data.append({'sequence': sequence, 'label': label})
    return pd.DataFrame(data)

# Load training data
train_df = load_data("sh_general_release_dynamic_19.02.2025.fasta")
X_train = train_df['sequence']
y_train = train_df['label']

# --- Feature Extraction (k-mer counts) ---
k = 6  # Optimize this parameter
vectorizer = CountVectorizer(analyzer='char', ngram_range=(k, k), lowercase=False)
X_train_counts = vectorizer.fit_transform(X_train)

# --- Model Training ---
model = MultinomialNB()
model.fit(X_train_counts, y_train)

# --- Test Data Processing ---
test_df = load_data("SRR33232122.fasta")
X_test = vectorizer.transform(test_df['sequence'])
y_test = test_df['label']

# --- Prediction and Evaluation ---
pred = model.predict(X_test)
mat = confusion_matrix(y_test, pred)

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(mat, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
