Below is the minimal, end-to-end PyTorch “Hello World” for a small regression task (predicting GC-content from a one-hot encoded 100-mer DNA sequence).

We’ll keep everything in a single file so you can copy-paste and run it immediately.

After the code you’ll find a block-by-block walk-through.

---

Code (save as `simple_torch.py`)

```python
# ------------- 1. Imports -------------
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ------------- 2. Toy data generator -------------
def random_dna(length=100):
    return ''.join(np.random.choice(list('ACGT'), length))

def gc(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)

class GCDataset(Dataset):
    def __init__(self, n_samples=1000):
        super().__init__()
        self.seqs = [random_dna() for _ in range(n_samples)]
        self.labels = [gc(s) for s in self.seqs]

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        x = one_hot(seq)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

def one_hot(seq):
    mapping = {'A':0, 'C':1, 'G':2, 'T':3}
    t = torch.zeros(4, len(seq))          # shape (4, 100)
    for i, base in enumerate(seq):
        t[mapping[base], i] = 1.0
    return t.flatten()                    # shape (400,)

# ------------- 3. Model -------------
class SimpleNet(nn.Module):
    def __init__(self, in_features=400):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze()      # squeeze removes extra dim

# ------------- 4. Training helpers -------------
def train(model, loader, optimizer, loss_fn):
    model.train()
    for x, y in loader:
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

@torch.no_grad()
def validate(model, loader, loss_fn):
    model.eval()
    losses = []
    for x, y in loader:
        pred = model(x)
        losses.append(loss_fn(pred, y).item())
    return np.mean(losses)

# ------------- 5. Run script -------------
if __name__ == "__main__":
    torch.manual_seed(42)

    # Data
    train_ds = GCDataset(800)
    val_ds   = GCDataset(200)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=32)

    # Model & training objects
    model = SimpleNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Loop
    for epoch in range(10):
        train(model, train_dl, optimizer, loss_fn)
        val_loss = validate(model, val_dl, loss_fn)
        print(f"Epoch {epoch+1:02d}  val_MSE={val_loss:.4f}")
```

Run it:

```bash
python simple_torch.py
```

---

Block-by-Block Explanation

1. Imports

   • `torch` – tensor engine.

   • `nn` – layers & loss functions.

   • `Dataset` / `DataLoader` – mini-batch streaming.

   • `numpy` – only for quick random DNA generation.

2. Toy data generator

   • `random_dna()` – returns a random 100-bp string.

   • `gc()` – ground-truth label (between 0 and 1).

   • `GCDataset` – PyTorch dataset:

     – `__len__` tells PyTorch how many samples.

     – `__getitem__` returns one (x, y) pair.

   • `one_hot()` – converts `"ACGT..."` → shape `(4, 100)` one-hot → flattens to `(400,)`.

3. Model (`SimpleNet`)

   • `nn.Sequential` stacks layers.

   • First `Linear` maps 400 inputs → 64 hidden units.

   • `ReLU` non-linearity.

   • Second `Linear` → 1 output (regression).

   • `forward()` describes the computation; `.squeeze()` removes the extra `(batch, 1)` dimension so `pred` matches `y`.

4. Training helpers

   • `train()` – one full pass over the loader.

     – `.zero_grad()` clears old gradients.

     – `loss.backward()` computes ∂loss/∂params.

     – `optimizer.step()` updates weights.

   • `@torch.no_grad()` decorator turns off gradient bookkeeping for validation.

5. Run script

   • Sets random seed for reproducibility.

   • Instantiates datasets & dataloaders.

   • Builds model, optimizer (Adam), loss (MSE).

   • 10-epoch loop prints validation MSE each epoch.

That’s the entire life-cycle: data → model → train → evaluate.

Once this runs, you can:

• swap `SimpleNet` for `nn.Conv1d` layers to keep spatial structure,

• change loss to `BCEWithLogitsLoss` for binary classification,

• or load real FASTA/FASTQ via `Biopython` instead of the toy generator.
