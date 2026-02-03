### Block 1: Installation and Import
PyTorch is the industry standard for Deep Learning research and production. You generally install it via pip or conda, checking the [PyTorch website](https://pytorch.org/) for the specific command for your CUDA version.

```python
# Install command (run in terminal):
# pip install torch torchvision torchaudio

import torch
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimization algorithms

print(f"PyTorch Version: {torch.__version__}")

# Check if GPU (CUDA) is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

### Block 2: Tensors (The Core Data Structure)
Tensors are similar to NumPy arrays but can run on GPUs to accelerate computing.

```python
# Creating Tensors
# 1. From data
data = [[1, 2], [3, 4]]
x_data = torch.tensor(data)

# 2. From NumPy array
import numpy as np
np_array = np.array(data)
x_np = torch.from_numpy(np_array)

# 3. Random or constant tensors
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor:\n {rand_tensor}")

# --- Operations ---

# 1. Device Management (Moving to GPU)
if torch.cuda.is_available():
    tensor = rand_tensor.to('cuda')
    print(f"Tensor is on GPU: {tensor.is_cuda}")

# 2. Standard NumPy-like indexing and slicing
tensor = torch.ones(4, 4)
print(f"First row: {tensor[0]}")
print(f"First column: {tensor[:, 0]}")
print(f"Last column: {tensor[..., -1]}")

# 3. Matrix Multiplication (@ operator or torch.matmul)
y1 = tensor @ tensor.T
y2 = tensor.matmul(tensor.T)

# 4. Element-wise product (* operator or torch.mul)
z1 = tensor * tensor
z2 = tensor.mul(tensor)
```

### Block 3: Autograd (Automatic Differentiation)
This is the engine that powers neural network training. It tracks operations on tensors to calculate gradients (derivatives) automatically.

```python
# Create a tensor and tell PyTorch to track its computation
x = torch.ones(2, 2, requires_grad=True)
print(f"Tensor with gradients: {x}")

# Perform an operation
y = x + 2
print(f"Result of x + 2: \n{y}")

# y was created as a result of an operation, so it has a grad_fn
print(f"Gradient function: {y.grad_fn}")

# Perform more operations
z = y * y * 3
out = z.mean()

print(f"Final Output: {out}")

# --- Backpropagation ---
# Calculate gradients: d(out)/dx
out.backward()

# Print gradients d(out)/dx
print(f"Gradients:\n{x.grad}")

# Example: Stopping gradient tracking
# Useful when you have a pretrained model and don't want to update it
x = torch.randn(2, requires_grad=True)
y = x * 2

with torch.no_grad():
    # Operations inside this block are not tracked for gradients
    z = y * 3
    print(f"Does z require grad? {z.requires_grad}") # False
```

### Block 4: Building a Neural Network (`nn.Module`)
All models in PyTorch must subclass `nn.Module`. You define layers in `__init__` and the flow of data in `forward`.

```python
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 1. Define layers
        # Flatten: 28x28 image -> 784 vector
        self.flatten = nn.Flatten()
        # Linear layer (Fully Connected): 784 input -> 512 output
        self.fc1 = nn.Linear(28*28, 512)
        # Output layer: 512 -> 10 (for 10 classes)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # 2. Define the forward pass
        x = self.flatten(x)
        # Apply ReLU activation function
        x = F.relu(self.fc1(x))
        # No activation on final layer (CrossEntropyLoss applies Softmax internally)
        x = self.fc2(x)
        return x

# Instantiate the model and move to GPU
model = SimpleNet().to(device)
print(model)
```

### Block 5: The Dataset and DataLoader
PyTorch provides `Dataset` and `DataLoader` to handle data efficiently, allowing for batching, shuffling, and parallel loading.

```python
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Assume we have some dummy data
# X: features (100 samples, 784 features), y: labels (100 samples)
X = torch.randn(100, 784)
y = torch.randint(0, 10, (100,))

# 1. Wrap tensors in a Dataset
dataset = TensorDataset(X, y)

# 2. Create a DataLoader
# batch_size: number of samples per gradient update
# shuffle: True for training data, False for test data
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Iterating through the DataLoader
print("Batch iteration:")
for batch_idx, (data, target) in enumerate(dataloader):
    print(f"Batch {batch_idx}: Data shape {data.shape}, Target shape {target.shape}")
    if batch_idx == 2: # Just print first 3 batches
        break
```

### Block 6: The Training Loop
This is the most critical block. It brings together the model, the loss function, the optimizer, and the data.

```python
# 1. Define Hyperparameters
learning_rate = 1e-3
batch_size = 64
epochs = 5

# 2. Initialize Model, Loss, and Optimizer
model = SimpleNet().to(device)

# Loss Function: CrossEntropy for multi-class classification
criterion = nn.CrossEntropyLoss()

# Optimizer: SGD (Stochastic Gradient Descent) or Adam
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 3. The Training Loop
def train_loop(dataloader, model, criterion, optimizer):
    size = len(dataloader.dataset)
    model.train() # Set model to training mode
    
    for batch, (X, y) in enumerate(dataloader):
        # Move data to device
        X, y = X.to(device), y.to(device)
        
        # 1. Compute prediction and loss
        pred = model(X)
        loss = criterion(pred, y)
        
        # 2. Backpropagation
        optimizer.zero_grad() # Clear old gradients
        loss.backward()       # Calculate new gradients
        optimizer.step()      # Update weights
        
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# Dummy data for demonstration
train_dataloader = DataLoader(TensorDataset(torch.randn(100, 1, 28, 28), torch.randint(0,10,(100,))), batch_size=16)

for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, criterion, optimizer)
print("Done!")
```

### Block 7: Convolutional Neural Networks (CNNs)
CNNs are the standard for image processing. They preserve spatial relationships using Convolutional layers.

```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Conv2d(in_channels, out_channels, kernel_size)
        self.conv1 = nn.Conv2d(1, 32, 3, 1) # Input 1 channel (grayscale), Output 32
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128) # Note: Input size depends on image dimensions
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv -> ReLU -> MaxPool
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully Connected layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        # Log Softmax for NLLLoss (or use CrossEntropyLoss directly)
        output = F.log_softmax(x, dim=1)
        return output

cnn_model = CNN().to(device)
# For a 28x28 input, this model expects specific dimensions
# print(cnn_model(torch.randn(1, 1, 28, 28))) # Test shape
```

### Block 8: Recurrent Neural Networks (RNNs / LSTMs)
RNNs are used for sequential data like time series or text.

```python
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM(input_size, hidden_size, num_layers)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        
        # Forward pass through LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# Example: Input is (Batch, Sequence Length, Input Size)
# e.g., Batch of 10, Sequence of 5 steps, each step has 20 features
lstm_model = LSTMModel(input_size=20, hidden_size=50, num_layers=2, num_classes=2).to(device)
sample_input = torch.randn(10, 5, 20).to(device)
output = lstm_model(sample_input)
print(f"LSTM Output Shape: {output.shape}")
```

### Block 9: Saving and Loading Models
There are two ways to save models: saving the whole model object (easier but less flexible) or saving just the `state_dict` (recommended).

```python
# 1. Save the model parameters (Recommended)
torch.save(model.state_dict(), 'model_weights.pth')

# 2. Load the model parameters
model = SimpleNet() # Need to instantiate the class again
model.load_state_dict(torch.load('model_weights.pth'))
model.to(device)
model.eval() # Set to evaluation mode

# 3. Saving a checkpoint (for resuming training)
# Useful to save optimizer state and epoch number too
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Loading a checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
```

### Block 10: Transfer Learning (Using Pretrained Models)
Instead of training from scratch, you can use models trained on massive datasets (like ImageNet).

```python
from torchvision import models

# 1. Load a pretrained ResNet18
pretrained_model = models.resnet18(pretrained=True)

# 2. Freeze the parameters (so we don't destroy the pretrained weights)
for param in pretrained_model.parameters():
    param.requires_grad = False

# 3. Modify the final fully connected layer
# Get the number of input features for the original FC layer
num_ftrs = pretrained_model.fc.in_features

# Replace it with a new layer for our specific number of classes (e.g., 2 classes)
pretrained_model.fc = nn.Linear(num_ftrs, 2)

# Only the parameters of the final layer are being optimized
optimizer = optim.SGD(pretrained_model.fc.parameters(), lr=0.001, momentum=0.9)

# Now you can train this model on your small dataset
print(pretrained_model.fc)
```

### Block 11: Evaluation and Testing
When testing, you must disable gradient calculation and set the model to eval mode to turn off Dropout and BatchNorm updates.

```python
def test_loop(dataloader, model, criterion):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    model.eval() # Set to evaluation mode
    with torch.no_grad(): # Disable gradient calculation
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += criterion(pred, y).item()
            
            # Calculate accuracy
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Create dummy test data
test_data = torch.randn(20, 1, 28, 28)
test_labels = torch.randint(0, 10, (20,))
test_dataset = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test_dataset, batch_size=4)

test_loop(test_loader, model, criterion)
```

### Block 12: Putting It Together (A Complete Workflow)
This block simulates a complete pipeline: Data -> Model -> Train -> Evaluate.

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# 1. Data Preparation
X_train = torch.randn(1000, 20) # 1000 samples, 20 features
y_train = torch.randint(0, 2, (1000,)) # Binary classification (0 or 1)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)

# 2. Model Definition
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid() # Sigmoid for binary probability output
        )
    def forward(self, x):
        return self.net(x)

model = BinaryClassifier()
criterion = nn.BCELoss() # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop
print("Starting Training...")
for epoch in range(3):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        # Forward
        y_pred = model(X_batch)
        loss = criterion(y_pred.squeeze(), y_batch.float())
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch+1} | Loss: {epoch_loss/len(train_loader):.4f}")

# 4. Inference (Prediction)
sample_input = torch.randn(1, 20)
with torch.no_grad():
    prediction = model(sample_input)
    print(f"\nPrediction for new sample: {prediction.item():.4f} (Class: {int(prediction.item() > 0.5)})")
```
