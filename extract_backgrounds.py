import cv2
import numpy as np
import random
from pathlib import Path

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

# Create output structure for background
for split in ["train", "valid", "test"]:
    (dataset_path / split / "background").mkdir(parents=True, exist_ok=True)

print("Extracting background patches from images...")
print()

total_extracted = 0
total_failed = 0

def get_random_crop(w, h, excl_boxes, size=224, max_attempts=50):
    for _ in range(max_attempts):
        if w <= size or h <= size:
            break
        
        x = random.randint(0, w - size)
        y = random.randint(0, h - size)
        crop_box = [x, y, x + size, y + size]
        
        # Check overlap
        overlap = False
        for ex in excl_boxes:
            # Overlap condition:
            if not (crop_box[2] <= ex[0] or crop_box[0] >= ex[2] or 
                    crop_box[3] <= ex[1] or crop_box[1] >= ex[3]):
                overlap = True
                break
                
        if not overlap:
            return crop_box
    return None

for appliance, class_name in appliances.items():
    appliance_path = original_data / appliance
    print(f"Processing {appliance} backgrounds...")

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

            # Read label to get exclusion boxes
            with open(label_file, 'r') as f:
                lines = f.readlines()

            if not lines:
                continue
                
            # Read image
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]
            
            excl_boxes = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                try:
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) == 4:
                        cx, cy, bw, bh = coords
                        x_min = int((cx - bw/2) * w)
                        y_min = int((cy - bh/2) * h)
                        x_max = int((cx + bw/2) * w)
                        y_max = int((cy + bh/2) * h)
                    elif len(coords) >= 8:
                        points = []
                        for i in range(0, min(len(coords), 8), 2):
                            px = int(coords[i] * w)
                            py = int(coords[i + 1] * h)
                            points.append([px, py])
                        points = np.array(points, dtype=np.int32)
                        x_min, y_min = points.min(axis=0)
                        x_max, y_max = points.max(axis=0)
                    else:
                        continue
                    
                    # Add buffer padding
                    pad = 20
                    excl_boxes.append([x_min - pad, y_min - pad, x_max + pad, y_max + pad])
                except Exception:
                    continue
                    
            crop_size = random.randint(150, 300)
            crop_box = get_random_crop(w, h, excl_boxes, size=crop_size)
            
            if crop_box:
                x1, y1, x2, y2 = crop_box
                cropped = img[y1:y2, x1:x2]
                
                output_file = (
                    dataset_path / split / "background" /
                    f"bg_{img_path.stem}{img_path.suffix}"
                )
                cv2.imwrite(str(output_file), cropped)
                extracted_count += 1
                total_extracted += 1
            else:
                total_failed += 1

        if extracted_count > 0:
            print(f"  {split:5s}: {extracted_count} backgrounds extracted")

print()
print(f"✓ Total background patches extracted: {total_extracted}")
print(f"✗ Failed (image too small or no safe spots): {total_failed}")
print()
