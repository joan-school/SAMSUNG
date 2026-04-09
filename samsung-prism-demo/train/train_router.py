"""
Train the Router MLP on GAP vectors extracted by Person B.

Prerequisites:
    data/gap_vectors_train.npy   shape: (N, 960)  float32
    data/gap_labels_train.npy    shape: (N,)       int64
        label 0 = Display expert  (TV images)
        label 1 = Kitchen expert  (Fridge + Microwave images)
        label 2 = Climate expert  (AC images)

Run:
    python train/train_router.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from src.router import RouterMLP

# ── Config ────────────────────────────────────────────────────────────
EPOCHS        = 50
BATCH_SIZE    = 64
LEARNING_RATE = 1e-3
VAL_SPLIT     = 0.15          # 15% of data held out for validation
SAVE_PATH     = "models/router.pt"

# ── Load data ─────────────────────────────────────────────────────────
print("Loading GAP vectors...")
X = np.load("data/gap_vectors_train.npy").astype(np.float32)
y = np.load("data/gap_labels_train.npy").astype(np.int64)

print(f"  Vectors shape : {X.shape}")     # expect (N, 960)
print(f"  Labels shape  : {y.shape}")     # expect (N,)
print(f"  Unique labels : {set(y)}")      # expect {{0, 1, 2}}

assert X.shape[1] == 960, "GAP vectors must be 960-dim!"
assert set(y).issubset({0, 1, 2}), "Labels must be 0, 1, or 2!"

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

dataset    = TensorDataset(X_tensor, y_tensor)
val_size   = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

print(f"  Train samples : {train_size}")
print(f"  Val samples   : {val_size}")

# ── Model, optimizer, loss ────────────────────────────────────────────
model     = RouterMLP(num_experts=3)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()

# ── Training loop ─────────────────────────────────────────────────────
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    # --- Train ---
    model.train()
    train_loss, train_correct = 0.0, 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss    += loss.item()
        train_correct += (logits.argmax(1) == yb).sum().item()

    train_acc = train_correct / train_size * 100

    # --- Validate ---
    model.eval()
    val_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            logits      = model(xb)
            val_correct += (logits.argmax(1) == yb).sum().item()

    val_acc = val_correct / val_size * 100

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Loss: {train_loss:.3f} | "
          f"Train Acc: {train_acc:.1f}% | "
          f"Val Acc: {val_acc:.1f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✅ Saved new best router (val acc: {val_acc:.1f}%)")

print(f"\nDone. Best val accuracy: {best_val_acc:.1f}%")
print(f"Router saved to: {SAVE_PATH}")