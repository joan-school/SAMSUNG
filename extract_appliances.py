"""
Extract appliances from images using YOLO labels and create cleaned dataset
"""

import cv2
import numpy as np
from pathlib import Path
import shutil

# Define appliances and source data
appliances = {
    "AIR CONDITIONER": "air_conditioner",
    "Microwave": "microwave",
    "Refrigerator": "refrigerator",
    "Television": "television"
}

base_path = Path("/Users/rakshithayathiraj/Desktop/files-2")
dataset_path = base_path / "dataset_cropped"
original_data = base_path

# Create output structure
for split in ["train", "valid", "test"]:
    for class_name in appliances.values():
        (dataset_path / split / class_name).mkdir(parents=True, exist_ok=True)

print("Extracting appliances from images...")
print()

total_extracted = 0
total_failed = 0

for appliance, class_name in appliances.items():
    appliance_path = original_data / appliance
    print(f"Processing {appliance}...")

    for split in ["train", "valid", "test"]:
        images_dir = appliance_path / split / "images"
        labels_dir = appliance_path / split / "labels"

        if not images_dir.exists():
            continue

        images = sorted(images_dir.glob("*"))
        extracted_count = 0

        for img_path in images:
            # Get label file
            base_name = img_path.stem
            label_file = labels_dir / f"{base_name}.txt"

            if not label_file.exists():
                continue

            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Read label (YOLO oriented format with polygon)
            with open(label_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                continue

            for line_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    coords = [float(x) for x in parts[1:]]

                    if len(coords) == 4:
                        # Regular YOLO format: center_x, center_y, width, height
                        cx, cy, bw, bh = coords
                        x_min = int((cx - bw/2) * w)
                        y_min = int((cy - bh/2) * h)
                        x_max = int((cx + bw/2) * w)
                        y_max = int((cy + bh/2) * h)
                    elif len(coords) >= 8:
                        # Oriented format with polygon (4 points)
                        points = []
                        for i in range(0, min(len(coords), 8), 2):
                            x = int(coords[i] * w)
                            y = int(coords[i + 1] * h)
                            points.append([x, y])

                        points = np.array(points, dtype=np.int32)
                        x_min, y_min = points.min(axis=0)
                        x_max, y_max = points.max(axis=0)
                    else:
                        continue

                    # Add padding
                    pad = 10
                    x_min = max(0, x_min - pad)
                    y_min = max(0, y_min - pad)
                    x_max = min(w, x_max + pad)
                    y_max = min(h, y_max + pad)

                    # Crop appliance
                    if x_max > x_min and y_max > y_min:
                        cropped = img[y_min:y_max, x_min:x_max]

                        # Save cropped image
                        output_file = (
                            dataset_path / split / class_name /
                            f"{img_path.stem}_{line_idx}{img_path.suffix}"
                        )
                        cv2.imwrite(str(output_file), cropped)
                        extracted_count += 1
                        total_extracted += 1

                except Exception as e:
                    total_failed += 1
                    continue

        if extracted_count > 0:
            print(f"  {split:5s}: {extracted_count} appliances extracted")

print()
print(f"✓ Total extracted: {total_extracted}")
print(f"✗ Failed: {total_failed}")
print()

# Show final count
print("Final cropped dataset structure:")
for split in ["train", "valid", "test"]:
    split_path = dataset_path / split
    total = 0
    for class_dir in sorted(split_path.iterdir()):
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*")))
            total += count
            print(f"  {split}/{class_dir.name}: {count} images")
    print(f"  {split} total: {total}")
    print()
