"""
Evaluate model on test set
"""

import torch
import torch.nn as nn
from pathlib import Path
from torchvision import datasets, transforms, models
import numpy as np

# CONFIG
MODEL_PATH = Path("outputs/mobilenetv3_finetuned_v2.pt")
TEST_DIR = Path("dataset_cropped/test")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps"
                      if torch.backends.mps.is_available() else "cpu")

# Load model
print(f"[INFO] Loading model...")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
classes = checkpoint["classes"]
num_classes = checkpoint["num_classes"]

model = models.mobilenet_v3_small(weights=None)
in_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.Hardswish(),
    nn.Dropout(p=0.3),
    nn.Linear(256, num_classes),
)
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(DEVICE)
model.eval()

# Load test data
tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

test_ds = datasets.ImageFolder(TEST_DIR, transform=tf)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

print(f"[INFO] Classes: {classes}")
print(f"[INFO] Test images: {len(test_ds)}")
print()

# Test
all_preds = []
all_labels = []
all_confidences = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confidence, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_confidences.extend(confidence.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)
all_confidences = np.array(all_confidences)

# Metrics
accuracy = (all_preds == all_labels).mean()
print(f"[RESULT] Test Accuracy: {accuracy*100:.2f}%")
print(f"[RESULT] Avg Confidence: {all_confidences.mean()*100:.2f}%")
print()

# Confusion matrix
print("Confusion Matrix:")
cm = np.zeros((num_classes, num_classes), dtype=int)
for true_label, pred_label in zip(all_labels, all_preds):
    cm[true_label][pred_label] += 1

# Print header
print("     ", end="")
for c in classes:
    print(f"{c:15s} ", end="")
print()

# Print rows
for i, class_name in enumerate(classes):
    print(f"{class_name:5s}", end=" ")
    for j in range(num_classes):
        print(f"{cm[i][j]:15d} ", end="")
    print()

print()

# Per-class accuracy
print("Per-class Results:")
for i, class_name in enumerate(classes):
    class_mask = all_labels == i
    if class_mask.sum() > 0:
        class_acc = (all_preds[class_mask] == i).mean()
        class_conf = all_confidences[class_mask].mean()
        total = class_mask.sum()
        print(f"  {class_name:20s} - Accuracy: {class_acc*100:5.1f}% | Confidence: {class_conf*100:5.1f}% | Images: {total}")

