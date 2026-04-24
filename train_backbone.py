"""
Fine-tunes MobileNetV3-Small on a Roboflow classification dataset
in two phases:
  Phase 1: freeze backbone, train classifier head  (~5 epochs)
  Phase 2: unfreeze all layers, fine-tune end-to-end (~10 epochs)

Saves:
  mobilenetv3_finetuned.pt  — best checkpoint (full model)
  training_log.json         — per-epoch metrics

Requirements:
  pip install torch torchvision roboflow tqdm
"""

import os, json, copy, time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR   = Path("./dataset")               # your local dataset folder
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE   = 224
BATCH_SIZE = 32
NUM_EPOCHS_PHASE1 = 5   # classifier head only
NUM_EPOCHS_PHASE2 = 10  # full backbone
LR_PHASE1  = 1e-3
LR_PHASE2  = 1e-4       # lower LR for fine-tuning
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps"
                      if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────────────────────────
#  STEP 2: Build DataLoaders
# ─────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 32, IMG_SIZE + 32)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_loaders():
    train_tf, val_tf = get_transforms()

    # Roboflow "folder" format puts images under train/valid/test folders
    train_dir = DATA_DIR / "train"
    val_dir   = DATA_DIR / "valid"

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=4, pin_memory=True)

    print(f"[INFO] Classes ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"[INFO] Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    return train_loader, val_loader, train_ds.classes


# ─────────────────────────────────────────────
#  STEP 3: Build model
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

    # Replace the classifier head for our number of classes
    # MobileNetV3-Small classifier: [Linear(576→1024), Hardswish, Dropout, Linear(1024→1000)]
    in_features = model.classifier[0].in_features  # 576
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


def freeze_backbone(model: nn.Module):
    """Freeze all layers except the classifier head."""
    for name, param in model.named_parameters():
        param.requires_grad = "classifier" in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Phase 1 — trainable params: {trainable:,}")


def unfreeze_all(model: nn.Module):
    """Unfreeze the full model for end-to-end fine-tuning."""
    for param in model.parameters():
        param.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[INFO] Phase 2 — trainable params: {trainable:,}")


# ─────────────────────────────────────────────
#  STEP 4: Training loop
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, phase="train"):
    is_train = phase == "train"
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, labels in tqdm(loader, desc=f"  {phase}", leave=False):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


def train(model, train_loader, val_loader, epochs, lr, phase_name, log):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc  = 0.0
    best_wts  = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, "train")
        va_loss, va_acc = run_epoch(model, val_loader,   criterion, optimizer, "val")
        scheduler.step()
        elapsed = time.time() - t0

        entry = {
            "phase": phase_name, "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss":   round(va_loss, 4), "val_acc":   round(va_acc, 4),
        }
        log.append(entry)
        flag = " ✓ best" if va_acc > best_acc else ""
        print(f"  [{phase_name}] Epoch {epoch:02d}/{epochs} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} | "
              f"{elapsed:.1f}s{flag}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    print(f"  [{phase_name}] Best val accuracy: {best_acc:.4f}")
    return model


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Using dataset from: {DATA_DIR}")
    
    if not DATA_DIR.exists():
        raise ValueError(f"Dataset directory not found: {DATA_DIR}")

    train_loader, val_loader, classes = build_loaders()
    num_classes = len(classes)

    model = build_model(num_classes)
    log   = []

    # ── Phase 1: head only ──
    print("\n[PHASE 1] Training classifier head (backbone frozen)...")
    freeze_backbone(model)
    model = train(model, train_loader, val_loader,
                  NUM_EPOCHS_PHASE1, LR_PHASE1, "phase1", log)

    # ── Phase 2: full fine-tune ──
    print("\n[PHASE 2] Fine-tuning full network...")
    unfreeze_all(model)
    model = train(model, train_loader, val_loader,
                  NUM_EPOCHS_PHASE2, LR_PHASE2, "phase2", log)

    # ── Save artefacts ──
    model_path = OUTPUT_DIR / "mobilenetv3_finetuned.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
    }, model_path)
    print(f"\n[INFO] Model saved → {model_path}")

    log_path = OUTPUT_DIR / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Training log → {log_path}")

    # Save class names for downstream scripts
    classes_path = OUTPUT_DIR / "classes.json"
    with open(classes_path, "w") as f:
        json.dump(classes, f, indent=2)
    print(f"[INFO] Class names → {classes_path}")


if __name__ == "__main__":
    main()
