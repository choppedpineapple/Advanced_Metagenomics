────────────────────────────────────────────
Part 0 – The 30-second executive summary
────────────────────────────────────────────
Every DL project boils down to the same four questions:

1. What is the raw datum?  
2. What tensor shape does my model expect?  
3. How do I build a deterministic, reversible map ① → ② ?  
4. How do I run that map quickly on thousands / millions of items?

If you can answer 1–3 for any modality, you can handle “every situation that arises”.

────────────────────────────────────────────
Part 1 – A mental framework (the “4×4×4” table)
────────────────────────────────────────────
Think of the problem as a 2×2×2 cube (hence 4×4×4 = 64 combinations):

          | 1-D               | 2-D                | 3-D               | n-D / graph

----------┼-------------------┼--------------------┼-------------------┼-------------
Discrete  | DNA seq, tokens   | AA k-mer grid      | 3-D voxel grid    | Graph nodes
Continuous| ChIP-seq signal   | Hi-C contact map   | Cryo-EM volume    | Point cloud
----------┼-------------------┼--------------------┼-------------------┼-------------
Fixed len | 100-mer           | 224×224 image      | 64×64×64 cube     | padded graph
Variable  | read length 50-300| microscopy 512×384 | tomogram slices   | diff #nodes

Your preprocessing pipeline is nothing more than a projection from your cell in the left half of the cube to the cell your model expects on the right half.

────────────────────────────────────────────
Part 2 – The three canonical patterns
────────────────────────────────────────────
Pattern A – One-hot / embedding for discrete sequences

Example: DNA, protein, SMILES strings.

```
"ACGT"  →  [4, len] one-hot  →  flatten or keep as (4, len) → Conv1d
```

Pattern B – Dense tensors for continuous signals

Example: coverage tracks, ChIP-seq, ATAC-seq.

```
[0.2, 0.0, 1.3, ...]  →  [1, len]  →  Conv1d or Transformer
```

Pattern C – Graph / set structures

Example: protein–protein interaction networks, scRNA-seq k-NN graphs.

```
nodes=[gene feats]  edges=[interactions]  →  PyG Data object
```

Your job is to recognise which pattern your data fits, then pick the minimal transform.

────────────────────────────────────────────
Part 3 – A reusable “DataAdapter” blueprint
────────────────────────────────────────────
Below is a single class that covers Pattern A & B for any sequence-like data (DNA, RNA, protein, k-mer counts, coverage tracks).

You can copy it once and reuse forever.

```python
import torch
from pathlib import Path
from typing import Callable, List
import numpy as np

class DataAdapter:
    """
    Universal adapter for 1-D discrete or continuous sequences.
    """
    def __init__(self,
                 alphabet: str = None,      # e.g. "ACGT" for DNA
                 fixed_len: int = None,     # crop / pad to this
                 transform: Callable = None # optional user fn
                 ):
        self.alphabet = alphabet
        self.fixed_len = fixed_len
        self.transform = transform
        
        # Build lookup tables once
        if alphabet:
            self.char2idx = {c: i for i, c in enumerate(alphabet)}
            self.vocab_size = len(alphabet)

    # ---------- PUBLIC API ----------
    def __call__(self, raw: str) -> torch.Tensor:
        """
        raw : str (discrete) OR List[float] (continuous)
        returns: torch tensor ready for model
        """
        # 1. Optional user transform
        if self.transform:
            raw = self.transform(raw)
        
        # 2. Crop / pad to fixed_len
        if self.fixed_len:
            raw = self._crop_or_pad(raw, self.fixed_len)
        
        # 3. Encode
        if self.alphabet:
            return self._to_onehot(raw)
        else:
            return torch.tensor(raw, dtype=torch.float32)

    # ---------- PRIVATE HELPERS ----------
    @staticmethod
    def _crop_or_pad(x, L):
        if isinstance(x, str):
            if len(x) > L:
                x = x[:L]
            else:
                x = x + "N" * (L - len(x))
        else:  # numeric list
            x = list(x)
            if len(x) > L:
                x = x[:L]
            else:
                x = x + [0.0] * (L - len(x))
        return x

    def _to_onehot(self, seq: str) -> torch.Tensor:
        t = torch.zeros(self.vocab_size, len(seq))
        for i, char in enumerate(seq):
            idx = self.char2idx.get(char.upper(), 0)  # default to 0 for 'N'
            t[idx, i] = 1.0
        return t.flatten()  # (vocab_size * len,)
```

Usage examples:

```python
# DNA 100-mers for a CNN
dna_adapter = DataAdapter(alphabet="ACGT", fixed_len=100)
tensor = dna_adapter("ACGTACGT...")        # shape (400,)

# Coverage track of length 1024
cov_adapter = DataAdapter(fixed_len=1024)  # no alphabet ⇒ numeric
tensor = cov_adapter([0.0, 1.3, 0.7, ...])  # shape (1024,)
```

────────────────────────────────────────────
Part 4 – Handling variable length & batching
────────────────────────────────────────────
CNNs and Transformers usually demand fixed length, RNNs & Transformers with attention masks do not.

Two idiomatic options:

Option 1 – Padding + collate_fn (works for everything)

```python
def collate_pad(batch):
    seqs, labels = zip(*batch)              # list of tensors, list of scalars
    seqs = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True)
    return seqs, torch.tensor(labels)
```

Option 2 – PackedSequence (RNNs)

```python
from torch.nn.utils.rnn import pack_padded_sequence

lengths = [len(s) for s in seqs]
padded = collate_pad(seqs)
packed = pack_padded_sequence(padded, lengths, enforce_sorted=False)
```

────────────────────────────────────────────
Part 5 – Graph data (Pattern C)
────────────────────────────────────────────
When the data is not a sequence but a set of objects with relations (protein–ligand complexes, regulatory networks), use PyTorch Geometric:

```python
from torch_geometric.data import Data

x = torch.tensor(node_features)      # [N, F]  (N nodes, F feats per node)
edge_index = torch.tensor(edges, dtype=torch.long)  # [2, E] (COO format)
data = Data(x=x, edge_index=edge_index)
```

Your “DataAdapter” for graphs simply returns a `Data` object instead of a tensor.

────────────────────────────────────────────
Part 6 – A mental checklist for any new modality
────────────────────────────────────────────
1. Describe raw datum in one sentence.  
2. Choose the nearest pattern (A/B/C).  
3. Decide fixed vs variable length.  
4. Write an invertible transform: raw → tensor.  
5. Benchmark two lines:  
   - `dataset[i]` returns correct tensor shape.  
   - `DataLoader(dataset, batch_size=32)` produces batches with `tensor.shape == (32, ...)`.

If both tests pass, you are done.

Everything else—normalisation, augmentation, masking, log1p-transform, etc.—is just additional steps inside the same adapter.

────────────────────────────────────────────
Recap cheat-sheet
────────────────────────────────────────────
• DNA / protein strings → one-hot → flatten or Conv1d.

• Numeric signals → float tensor → Conv1d or Transformer.

• Sets / graphs → PyG Data object → GNN layers.

• Always wrap the transform in a reusable class; keep the model code agnostic to raw data.
