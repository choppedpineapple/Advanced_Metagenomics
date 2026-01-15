"""
Genomics + PyTorch learning script
---------------------------------
This file is intentionally "block-based" with clear sections and long-form
comments. It is meant for beginners who want to learn deep learning by doing.

Goal: build a tiny DNA sequence classifier that detects a motif (a short DNA
pattern) inside random sequences. We use synthetic data to keep things simple.
"""

# ---------------------------------------------------------------------------
# Block 0: Bird's eye overview (read this before the code)
# ---------------------------------------------------------------------------
# 1) Create synthetic DNA sequences with labels:
#    - Positive class: contains a known motif (e.g. "TATAAA")
#    - Negative class: random sequence without forced motif insertion
# 2) Encode DNA letters (A, C, G, T) into numbers (one-hot vectors).
# 3) Build a simple neural network (1D CNN) that can "scan" the sequence.
# 4) Train the model with labeled examples.
# 5) Evaluate and run inference on new sequences.
#
# This mirrors a real bioinformatics workflow:
#   raw sequences -> numerical encoding -> model -> prediction

# ---------------------------------------------------------------------------
# Block 1: Imports and reproducibility
# ---------------------------------------------------------------------------
# Imports are at the top so Python can load required libraries.
import random
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


# Set seeds so results are repeatable. This is helpful when learning.
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Choose a device: GPU if available, otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Block 2: DNA helpers (generate synthetic data)
# ---------------------------------------------------------------------------
# DNA uses the alphabet A, C, G, T. We'll build sequences from these letters.
DNA_ALPHABET = ["A", "C", "G", "T"]


def make_random_dna(length: int) -> str:
    """Return a random DNA string of a given length."""
    return "".join(random.choice(DNA_ALPHABET) for _ in range(length))


def insert_motif(sequence: str, motif: str) -> str:
    """
    Insert a motif at a random position inside the sequence.
    This guarantees the motif appears at least once.
    """
    if len(motif) > len(sequence):
        raise ValueError("Motif longer than sequence.")
    pos = random.randint(0, len(sequence) - len(motif))
    return sequence[:pos] + motif + sequence[pos + len(motif) :]


def build_dataset(
    num_samples: int, seq_length: int, motif: str, positive_ratio: float
) -> Tuple[List[str], List[int]]:
    """
    Create a synthetic dataset of DNA strings and labels.
    - label 1: motif is inserted (positive class)
    - label 0: random sequence (negative class)
    """
    sequences: List[str] = []
    labels: List[int] = []

    for _ in range(num_samples):
        is_positive = random.random() < positive_ratio
        seq = make_random_dna(seq_length)
        if is_positive:
            seq = insert_motif(seq, motif)
            label = 1
        else:
            label = 0

        sequences.append(seq)
        labels.append(label)

    return sequences, labels


# ---------------------------------------------------------------------------
# Block 3: Encoding DNA as numbers (one-hot encoding)
# ---------------------------------------------------------------------------
# Deep learning models only work with numbers, so we encode DNA letters.
# One-hot encoding: each letter becomes a 4-length vector.
#
# A -> [1, 0, 0, 0]
# C -> [0, 1, 0, 0]
# G -> [0, 0, 1, 0]
# T -> [0, 0, 0, 1]

DNA_TO_INDEX = {"A": 0, "C": 1, "G": 2, "T": 3}


def one_hot_encode(sequence: str) -> torch.Tensor:
    """
    Convert a DNA string into a tensor of shape (4, length).
    - 4 rows (A, C, G, T)
    - length columns (positions in the sequence)
    """
    length = len(sequence)
    tensor = torch.zeros(4, length, dtype=torch.float32)
    for i, base in enumerate(sequence):
        tensor[DNA_TO_INDEX[base], i] = 1.0
    return tensor


# ---------------------------------------------------------------------------
# Block 4: Dataset class (PyTorch-friendly)
# ---------------------------------------------------------------------------
# PyTorch uses Dataset objects to represent data. Each item is a (x, y) pair.


class DnaMotifDataset(Dataset):
    """Synthetic DNA dataset that returns (one_hot_tensor, label)."""

    def __init__(
        self, num_samples: int, seq_length: int, motif: str, positive_ratio: float
    ) -> None:
        self.sequences, self.labels = build_dataset(
            num_samples, seq_length, motif, positive_ratio
        )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]
        x = one_hot_encode(seq)
        # BCEWithLogitsLoss expects float labels (0.0 or 1.0).
        y = torch.tensor(label, dtype=torch.float32)
        return x, y


# ---------------------------------------------------------------------------
# Block 5: Model definition (simple 1D CNN)
# ---------------------------------------------------------------------------
# A 1D convolution slides a small window across the sequence, which is perfect
# for detecting patterns like motifs.


class SimpleDnaCNN(nn.Module):
    """A tiny CNN for motif detection."""

    def __init__(self, motif_length: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=4, out_channels=8, kernel_size=motif_length
        )
        self.relu = nn.ReLU()
        # Global max pooling: keep the strongest motif signal anywhere.
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 4, length)
        x = self.conv(x)    # -> (batch, 8, length - motif_length + 1)
        x = self.relu(x)
        x = self.pool(x)    # -> (batch, 8, 1)
        x = x.squeeze(-1)   # -> (batch, 8)
        x = self.fc(x)      # -> (batch, 1)
        return x.squeeze(-1)  # -> (batch,)


# ---------------------------------------------------------------------------
# Block 6: Training utilities
# ---------------------------------------------------------------------------


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Convert logits to probabilities, then to class predictions, then compute
    accuracy.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    correct = (preds == labels).sum().item()
    return correct / len(labels)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    is_train: bool,
) -> Tuple[float, float]:
    """
    Run one epoch. If is_train is True, update weights. Otherwise just evaluate.
    """
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0

    for x, y in data_loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        logits = model(x)
        loss = loss_fn(logits, y)
        acc = accuracy_from_logits(logits, y)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += acc

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / len(data_loader)
    return avg_loss, avg_acc


# ---------------------------------------------------------------------------
# Block 7: Main training script
# ---------------------------------------------------------------------------


def main() -> None:
    # Hyperparameters are tunable knobs. These are simple starter values.
    seq_length = 50
    motif = "TATAAA"
    positive_ratio = 0.5
    num_samples = 2000
    batch_size = 32
    epochs = 6
    learning_rate = 1e-3

    # 1) Create dataset
    dataset = DnaMotifDataset(
        num_samples=num_samples,
        seq_length=seq_length,
        motif=motif,
        positive_ratio=positive_ratio,
    )

    # 2) Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 3) Build model and training tools
    model = SimpleDnaCNN(motif_length=len(motif)).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 4) Train loop
    print(f"Device: {DEVICE}")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, loss_fn, optimizer, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, loss_fn, optimizer, is_train=False
        )

        print(
            f"Epoch {epoch:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

    # 5) Inference example
    print("\nInference demo:")
    test_seq = insert_motif(make_random_dna(seq_length), motif)
    test_x = one_hot_encode(test_seq).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logit = model(test_x)
        prob = torch.sigmoid(logit).item()
    print(f"Sequence: {test_seq}")
    print(f"Predicted probability of motif: {prob:.3f}")


# ---------------------------------------------------------------------------
# Block 8: Script entry point
# ---------------------------------------------------------------------------
# This is the standard Python pattern for scripts. It lets you import this file
# without running main(), but running it directly will execute main().
if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# Block 9: Learning extensions (ideas for you)
# ---------------------------------------------------------------------------
# 1) Replace the synthetic dataset with real DNA sequences (FASTA files).
# 2) Change the model to an RNN or transformer for longer sequences.
# 3) Add more classes (e.g., multiple motifs).
# 4) Track precision/recall if classes become imbalanced.
# 5) Plot training curves to visualize learning progress.
