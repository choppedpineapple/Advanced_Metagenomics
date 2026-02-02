
### Block 1: Imports and Hardware Setup
**Goal:** Import the necessary PyTorch libraries and check if a GPU is available. Deep learning is much faster on GPUs.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Check if GPU (CUDA) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

---

### Block 2: Hyperparameters
**Goal:** Define the settings that control the behavior of the training process. Keeping these at the top makes it easy to tweak the model later.

```python
# Hyperparameters
BATCH_SIZE = 64        # How many images to process at once
LEARNING_RATE = 0.001  # How big of a step the optimizer takes
EPOCHS = 5             # How many times to loop over the entire dataset
INPUT_SIZE = 28 * 28   # MNIST images are 28x28 pixels
HIDDEN_SIZE = 128      # Number of neurons in the hidden layer
NUM_CLASSES = 10       # Digits 0-9
```

---

### Block 3: Data Preparation (Transforms and Dataset)
**Goal:** Download the data and turn images into Tensors. We need to normalize the data (scale pixel values from 0-255 to 0-1) so the math works better.

```python
# Define a transform to convert images to tensors and normalize them
transform = transforms.Compose([
    transforms.ToTensor(), # Converts PIL image to PyTorch Tensor (C x H x W) in range [0, 1]
    transforms.Normalize((0.5,), (0.5,)) # Normalizes mean and std dev to 0.5 (range becomes [-1, 1])
])

# Download and load training data
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Download and load test data
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

print("Data downloaded successfully.")
```

---

### Block 4: DataLoaders
**Goal:** Wrap the datasets in a DataLoader. The DataLoader handles batching (grouping images), shuffling (randomizing order for training), and loading data in parallel.

```python
# Create DataLoaders
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Check one batch to see the shape
images, labels = next(iter(train_loader))
print(f"Image batch shape: {images.shape}") # [64, 1, 28, 28] -> (Batch, Channel, Height, Width)
print(f"Labels batch shape: {labels.shape}")
```

---

### Block 5: Define the Neural Network Model
**Goal:** Create the class that defines the architecture.
*   **`__init__`**: Define the layers.
*   **`forward`**: Define how data flows through those layers.
*   We flatten the 2D image into a 1D vector to feed it into linear layers.

```python
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        # Layer 1: Input to Hidden
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Activation Function: ReLU (Rectified Linear Unit)
        self.relu = nn.ReLU()
        # Layer 2: Hidden to Output
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Flatten the image: (Batch, 1, 28, 28) -> (Batch, 784)
        x = x.view(-1, input_size) 
        
        # Pass through layers
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        
        # Note: We do not apply Softmax here because CrossEntropyLoss does it internally.
        return out

# Initialize the model and move it to the GPU/CPU
model = SimpleNet(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)
print(model)
```

---

### Block 6: Loss Function and Optimizer
**Goal:** Define how the model learns.
*   **Loss Function**: Calculates how wrong the model's predictions are (`CrossEntropyLoss` is standard for multi-class classification).
*   **Optimizer**: Updates the weights based on the loss (`Adam` is a great general-purpose optimizer).

```python
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
```

---

### Block 7: The Training Loop
**Goal:** This is the engine. We iterate through the data, make predictions, calculate error, and update weights.

```python
# Training Loop
for epoch in range(EPOCHS):
    model.train() # Set model to training mode (important for dropout/batchnorm)
    
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        # Move data to the device (GPU/CPU)
        images = images.to(device)
        labels = labels.to(device)
        
        # 1. Clear previous gradients (zero_grad)
        optimizer.zero_grad()
        
        # 2. Forward pass: Compute predicted outputs
        outputs = model(images)
        
        # 3. Compute loss
        loss = criterion(outputs, labels)
        
        # 4. Backward pass: Compute gradient of the loss
        loss.backward()
        
        # 5. Optimizer step: Update parameters
        optimizer.step()
        
        running_loss += loss.item()
        
    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")
```

---

### Block 8: The Evaluation Loop
**Goal:** Test the model on data it has never seen before to check accuracy.
**Crucial:** We use `torch.no_grad()` to stop PyTorch from calculating gradients here, which saves memory and speeds things up.

```python
# Evaluation
model.eval() # Set model to evaluation mode
with torch.no_grad(): # Disable gradient calculation
    n_correct = 0
    n_samples = 0
    
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Get the index of the max log-probability (the predicted class)
        # value, index = torch.max(input, dim)
        _, predicted = torch.max(outputs, 1)
        
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f"Accuracy on the 10,000 test images: {acc:.2f}%")
```

---

### Block 9: Making a Single Prediction (Inference)
**Goal:** Take a single image from the test set and run it through the model to see what it predicts.

```python
# Let's look at the first image of the test set
single_image, true_label = test_dataset[0]

# Add a batch dimension (model expects 4D input: Batch, Channel, Height, Width)
# (1, 28, 28) -> (1, 1, 28, 28)
single_image = single_image.unsqueeze(0).to(device)

# Predict
model.eval()
with torch.no_grad():
    output = model(single_image)
    _, prediction = torch.max(output, 1)

print(f"True Label: {true_label}")
print(f"Predicted Label: {prediction.item()}")
```

### Summary of Key Concepts Learned
1.  **Device Management**: Moving tensors between CPU and GPU (`.to(device)`).
2.  **DataLoader**: Handling batching and shuffling automatically.
3.  **Model Class**: Inheriting from `nn.Module` and defining `forward`.
4.  **The 5-Step Loop**:
    1.  `optimizer.zero_grad()`
    2.  `output = model(inputs)`
    3.  `loss = criterion(output, target)`
    4.  `loss.backward()`
    5.  `optimizer.step()`
5.  **Evaluation**: Using `torch.no_grad()` and `model.eval()` for testing.
