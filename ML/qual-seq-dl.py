import torch
import torch.nn as nn
import gzip
import random
from collections import Counter

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# === DATA PROCESSING ===
def read_fastq(file_path):
    """Read sequences and quality scores from FASTQ file"""
    sequences = []
    qualities = []
    
    # Handle both .fastq and .fastq.gz files
    if file_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    with open_func(file_path, mode) as f:
        while True:
            header = f.readline()
            if not header:
                break
            sequence = f.readline().strip()
            f.readline()  # Skip + line
            quality = f.readline().strip()
            
            sequences.append(sequence)
            qualities.append(quality)
    
    return sequences, qualities

def quality_to_phred(quality_str):
    """Convert ASCII quality to Phred scores"""
    return [ord(char) - 33 for char in quality_str]

def sequence_to_onehot(sequence):
    """Convert DNA sequence to one-hot encoding"""
    mapping = {'A': [1, 0, 0, 0],
               'T': [0, 1, 0, 0],
               'C': [0, 0, 1, 0],
               'G': [0, 0, 0, 1],
               'N': [0, 0, 0, 0]}  # Ambiguous base
    
    encoded = []
    for base in sequence[:50]:  # Limit to first 50 bases
        encoded.extend(mapping.get(base, [0, 0, 0, 0]))
    return encoded

# === MODEL DEFINITION ===
class DnaQualityClassifier(nn.Module):
    def __init__(self, input_size=200, hidden_size=64):
        super(DnaQualityClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)  # 2 classes: high/low quality
        )
    
    def forward(self, x):
        return self.network(x)

# === DATA PREPARATION ===
def prepare_data(sequences, qualities, threshold=30):
    """Convert raw data to model-ready tensors"""
    X = []
    y = []
    
    for seq, qual in zip(sequences, qualities):
        # Convert sequence to numbers
        seq_encoded = sequence_to_onehot(seq)
        
        # Pad or truncate to fixed length
        if len(seq_encoded) < 200:
            seq_encoded.extend([0] * (200 - len(seq_encoded)))
        else:
            seq_encoded = seq_encoded[:200]
        
        # Calculate average quality
        phred_scores = quality_to_phred(qual)
        avg_quality = sum(phred_scores) / len(phred_scores)
        
        # Create label (0=low quality, 1=high quality)
        label = 1 if avg_quality >= threshold else 0
        
        X.append(seq_encoded)
        y.append(label)
    
    return torch.FloatTensor(X), torch.LongTensor(y)

# === TRAINING FUNCTION ===
def train_model(model, train_loader, epochs=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.2f}%")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Replace these paths with your actual FASTQ files
    train_file = "train.fastq.gz"  # Training data
    test_file = "test.fastq.gz"    # Testing data
    
    # 1. Load training data
    print("Loading training data...")
    train_sequences, train_qualities = read_fastq(train_file)
    
    # 2. Prepare training tensors
    print("Preparing training data...")
    X_train, y_train = prepare_data(train_sequences, train_qualities)
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # 3. Initialize model
    model = DnaQualityClassifier().to(device)
    print("Model initialized")
    
    # 4. Train model
    print("Starting training...")
    train_model(model, train_loader, epochs=15)
    
    # 5. Evaluate on test data
    print("Loading test data...")
    test_sequences, test_qualities = read_fastq(test_file)
    X_test, y_test = prepare_data(test_sequences, test_qualities)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
    
    # 6. Test model
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f"Test Accuracy: {100 * correct / total:.2f}%")
