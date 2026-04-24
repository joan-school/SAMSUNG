"""
Loads the fine-tuned MobileNetV3-Small, strips the classifier head,
and runs inference over train + val sets to extract Global Average
Pooling (GAP) vectors (576-dim) as .npy files.


Outputs (in ./outputs/gap_vectors/):
  train_features.npy   shape: (N_train, 576)
  train_labels.npy     shape: (N_train,)
  val_features.npy     shape: (N_val, 576)
  val_labels.npy       shape: (N_val,)
  class_names.json

Requirements:
  pip install torch torchvision numpy tqdm
"""

import json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
OUTPUT_DIR  = Path("./outputs")
GAP_DIR     = OUTPUT_DIR / "gap_vectors"
MODEL_PATH  = OUTPUT_DIR / "mobilenetv3_finetuned.pt"
DATA_DIR    = Path("./dataset")

IMG_SIZE    = 224
BATCH_SIZE  = 64
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "mps"
                           if torch.backends.mps.is_available() else "cpu")


# ─────────────────────────────────────────────
#  Build backbone (no classifier head)
# ─────────────────────────────────────────────
class MobileNetV3Backbone(nn.Module):
    """
    MobileNetV3-Small with the classifier removed.
    Returns the 576-dim GAP vector (output of adaptive_avg_pool2d,
    after the features extractor).
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        # features = all conv/bn/activation blocks
        self.features = base_model.features
        # avgpool = AdaptiveAvgPool2d that produces the GAP vector
        self.avgpool  = base_model.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)   # (B, 576, H', W')
        x = self.avgpool(x)    # (B, 576, 1, 1)
        x = x.flatten(1)       # (B, 576)
        return x


def load_backbone() -> tuple[nn.Module, list]:
    print(f"[INFO] Loading model from {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes    = checkpoint["classes"]
    num_classes = checkpoint["num_classes"]

    # Rebuild the full model architecture
    full_model = models.mobilenet_v3_small(weights=None)
    in_features = full_model.classifier[0].in_features
    full_model.classifier = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    full_model.load_state_dict(checkpoint["model_state_dict"])

    # Strip the head — keep only features + avgpool
    backbone = MobileNetV3Backbone(full_model).to(DEVICE)
    backbone.eval()

    # Quick sanity check on output shape
    dummy   = torch.zeros(2, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)
    with torch.no_grad():
        out = backbone(dummy)
    assert out.shape == (2, 576), f"Unexpected GAP shape: {out.shape}"
    print(f"[INFO] Backbone ready — GAP dim: {out.shape[1]}")
    return backbone, classes


# ─────────────────────────────────────────────
#  DataLoaders (no augmentation for extraction)
# ─────────────────────────────────────────────
def build_loader(split: str) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(DATA_DIR / split, transform=tf)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=4, pin_memory=True)


# ─────────────────────────────────────────────
#  Extraction
# ─────────────────────────────────────────────
@torch.no_grad()
def extract_features(backbone: nn.Module, loader: DataLoader,
                     split: str) -> tuple[np.ndarray, np.ndarray]:
    all_feats, all_labels = [], []
    for images, labels in tqdm(loader, desc=f"  Extracting [{split}]"):
        images = images.to(DEVICE)
        feats  = backbone(images).cpu().numpy()
        all_feats.append(feats)
        all_labels.append(labels.numpy())
    feats  = np.concatenate(all_feats,  axis=0)
    labels = np.concatenate(all_labels, axis=0)
    print(f"  [{split}] features: {feats.shape}  labels: {labels.shape}")
    return feats, labels


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    GAP_DIR.mkdir(parents=True, exist_ok=True)

    backbone, classes = load_backbone()

    for split in ("train", "valid"):
        # Roboflow uses "valid" folder name
        folder_name = split
        split_label = "val" if split == "valid" else "train"

        loader = build_loader(folder_name)
        feats, labels = extract_features(backbone, loader, split_label)

        feat_path  = GAP_DIR / f"{split_label}_features.npy"
        label_path = GAP_DIR / f"{split_label}_labels.npy"
        np.save(feat_path,  feats)
        np.save(label_path, labels)
        print(f"  Saved → {feat_path}")
        print(f"  Saved → {label_path}")

    # Also extract test split if present
    test_dir = DATA_DIR / "test"
    if test_dir.exists():
        loader = build_loader("test")
        feats, labels = extract_features(backbone, loader, "test")
        np.save(GAP_DIR / "test_features.npy", feats)
        np.save(GAP_DIR / "test_labels.npy",   labels)
        print(f"  Saved → {GAP_DIR / 'test_features.npy'}")

    # Save class names
    with open(GAP_DIR / "class_names.json", "w") as f:
        json.dump(classes, f, indent=2)
    print(f"  Saved → {GAP_DIR / 'class_names.json'}")

    # ── Verification summary ──
    print("\n[VERIFICATION]")
    tr_f = np.load(GAP_DIR / "train_features.npy")
    tr_l = np.load(GAP_DIR / "train_labels.npy")
    print(f"  train_features : {tr_f.shape}  dtype={tr_f.dtype}  "
          f"min={tr_f.min():.3f}  max={tr_f.max():.3f}")
    print(f"  train_labels   : {tr_l.shape}  unique={np.unique(tr_l).tolist()}")
    print(f"  classes        : {classes}")
    print("\n[DONE] Hand off gap_vectors/ folder to Person A for router MLP training.")


if __name__ == "__main__":
    main()
