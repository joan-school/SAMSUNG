"""
Fine-tune MobileNetV3-Small on cropped appliance dataset with aggressive augmentation
"""

import json
import copy
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR = Path("./dataset_cropped")
OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 20  # More epochs for better convergence
LR = 5e-4  # Lower learning rate
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps"
                      if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────────────────────────
#  AGGRESSIVE DATA AUGMENTATION
# ─────────────────────────────────────────────
def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 64, IMG_SIZE + 64)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(45),  # More rotation
        transforms.ColorJitter(
            brightness=0.4,  # More brightness variation
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        ),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.8, 1.2),
            shear=15
        ),
        transforms.RandomPerspective(p=0.3, distortion_scale=0.3),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.3)),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_tf, val_tf


def build_loaders():
    train_tf, val_tf = get_transforms()

    train_ds = datasets.ImageFolder(DATA_DIR / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(DATA_DIR / "valid", transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"[INFO] Classes ({len(train_ds.classes)}): {train_ds.classes}")
    print(f"[INFO] Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    return train_loader, val_loader, train_ds.classes, train_ds


# ─────────────────────────────────────────────
#  BUILD MODEL
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.4),  # Increased dropout
        nn.Linear(256, num_classes),
    )
    return model.to(DEVICE)


# ─────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, phase="train", scheduler=None):
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    if is_train and scheduler:
        scheduler.step()

    return avg_loss, accuracy


def train(model, train_loader, val_loader, num_classes, log, train_ds):
    targets = [s[1] for s in train_ds.samples]
    counts = [targets.count(i) for i in range(num_classes)]
    weights = [1.0 / c if c > 0 else 0.0 for c in counts]
    sum_w = sum(weights)
    weights = [w / sum_w * num_classes for w in weights]
    class_weights = torch.FloatTensor(weights).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5,
        T_mult=1,
        eta_min=1e-6
    )

    best_acc = 0.0
    best_wts = copy.deepcopy(model.state_dict())
    patience_counter = 0
    patience = 5

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(
            model, train_loader, criterion, optimizer, "train", scheduler
        )
        va_loss, va_acc = run_epoch(
            model, val_loader, criterion, optimizer, "val"
        )
        elapsed = time.time() - t0

        entry = {
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_acc": round(va_acc, 4),
        }
        log.append(entry)

        flag = " ⭐ best" if va_acc > best_acc else ""
        print(
            f"  Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.3f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.3f} | "
            f"{elapsed:.1f}s{flag}"
        )

        if va_acc > best_acc:
            best_acc = va_acc
            best_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"  [EARLY STOPPING] No improvement for {patience} epochs")
            break

    model.load_state_dict(best_wts)
    print(f"  Best validation accuracy: {best_acc:.4f}")
    return model, best_acc


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Using dataset from: {DATA_DIR}")
    print()

    if not DATA_DIR.exists():
        raise ValueError(f"Dataset directory not found: {DATA_DIR}")

    train_loader, val_loader, classes, train_ds = build_loaders()
    num_classes = len(classes)
    print()

    model = build_model(num_classes)
    log = []

    print("[TRAINING] Fine-tuning with cropped appliances + aggressive augmentation...")
    print()

    model, best_acc = train(model, train_loader, val_loader, num_classes, log, train_ds)

    # ── Save artefacts ──
    model_path = OUTPUT_DIR / "mobilenetv3_finetuned_v2.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": classes,
        "num_classes": num_classes,
        "img_size": IMG_SIZE,
    }, model_path)
    print(f"\n[INFO] Model saved → {model_path}")

    log_path = OUTPUT_DIR / "training_log_v2.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"[INFO] Training log → {log_path}")

    print(f"\n[DONE] Best validation accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
