import torch
import torch.nn as nn
import torch.optim as optim

# ---- Utility: one-hot encode DNA ----
def one_hot_encode(seq):
    mapping = {"A":0, "C":1, "G":2, "T":3}
    encoding = torch.zeros(len(seq), 4)  # (length, 4)
    for i, base in enumerate(seq):
        encoding[i, mapping[base]] = 1
    return encoding.view(-1)  # flatten into 1D vector

# ---- Sample dataset (tiny for demo) ----
sequences = ["ATAT", "CGCG", "AAAA", "GGCC", "TTAA", "GCGC"]
labels    = [0,      1,      0,      1,      0,      1]  # 0=AT-rich, 1=GC-rich

# Convert to tensors
x = torch.stack([one_hot_encode(seq) for seq in sequences])  # shape (N, 16)
y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)   # shape (N, 1)

# ---- Define model ----
model = nn.Sequential(
    nn.Linear(16, 8),  # input=16 (4 bases * 4 letters), hidden=8
    nn.ReLU(),
    nn.Linear(8, 1),   # output=1
    nn.Sigmoid()       # probability
)

# ---- Loss & optimizer ----
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ---- Training loop ----
for epoch in range(200):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---- Test model ----
test_seq = "ATGC"
test_x = one_hot_encode(test_seq).unsqueeze(0)  # shape (1,16)
prediction = model(test_x).item()
print(f"Sequence: {test_seq}, GC-rich probability: {prediction:.2f}")
