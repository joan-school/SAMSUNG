"""
Live appliance detection using the trained MobileNetV3 model.

Usage:
  python3 live_detection.py              # Uses webcam (camera 0)
  python3 live_detection.py --video path/to/video.mp4  # Uses video file
  python3 live_detection.py --image path/to/image.jpg  # Single image
"""

import argparse
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torchvision import transforms, models

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
MODEL_PATH = Path("outputs/mobilenetv3_finetuned_v2.pt")
CLASSES_PATH = Path("outputs/classes.json")
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps"
                      if torch.backends.mps.is_available() else "cpu")

# Colors for different classes (BGR for OpenCV)
COLORS = {
    "air_conditioner": (0, 165, 255),    # Orange
    "microwave": (0, 255, 0),             # Green
    "refrigerator": (255, 0, 0),          # Blue
    "television": (255, 0, 255),          # Magenta
}

# ─────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────
def load_model():
    print(f"[INFO] Loading model from {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    num_classes = checkpoint["num_classes"]
    classes = checkpoint["classes"]

    # Build base model
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

    print(f"[INFO] Model loaded. Classes: {classes}")
    return model, classes


def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def predict(frame, model, transform):
    """Get prediction and confidence for frame"""
    # Convert BGR to RGB for torchvision model
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize for model
    img_tensor = transform(frame_rgb).unsqueeze(0).to(DEVICE)

    # Predict
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)
    confidence, pred_class = torch.max(probs, dim=1)

    return pred_class.item(), confidence.item()

def draw_target_box_and_crop(frame, box_size=400):
    """Calculates and extracts the central targeting square."""
    h, w = frame.shape[:2]
    size = min(box_size, h - 20, w - 20)
    x1 = (w - size) // 2
    y1 = (h - size) // 2
    x2 = x1 + size
    y2 = y1 + size
    crop = frame[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def run_webcam():
    """Live detection from webcam"""
    model, classes = load_model()
    transform = get_transform()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("[ERROR] Cannot open webcam")
        return

    print("[INFO] Starting webcam detection. Press 'q' to quit.")
    fps_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop and get coordinates
        crop, (x1, y1, x2, y2) = draw_target_box_and_crop(frame)

        # Predict
        class_idx, confidence = predict(crop, model, transform)
        class_name = classes[class_idx]
        
        if confidence < 0.85 or class_name == "background":
            text = "BACKGROUND"
            color = (150, 150, 150)
        else:
            text = f"{class_name.upper()}: {confidence*100:.1f}%"
            color = COLORS.get(class_name, (255, 255, 255))

        # Draw Green Target Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text above the box
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 2)

        # FPS counter
        fps_counter += 1
        if fps_counter % 10 == 0:
            print(f"[DETECTION] {class_name} ({confidence*100:.1f}%)")

        # Display
        cv2.imshow("Live Appliance Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam detection ended")


def run_video(video_path):
    """Live detection from video file"""
    model, classes = load_model()
    transform = get_transform()

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    print(f"[INFO] Starting video detection: {video_path}")
    print("[INFO] Press 'q' to quit or wait for video to end.")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Crop and get coordinates
        crop, (x1, y1, x2, y2) = draw_target_box_and_crop(frame)

        # Predict
        class_idx, confidence = predict(crop, model, transform)
        class_name = classes[class_idx]
        
        if confidence < 0.85 or class_name == "background":
            text = "BACKGROUND"
            color = (150, 150, 150)
        else:
            text = f"{class_name.upper()}: {confidence*100:.1f}%"
            color = COLORS.get(class_name, (255, 255, 255))

        # Draw Green Target Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add text above the box
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, color, 2)

        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        print(f"[FRAME {frame_count}] {class_name} ({confidence*100:.1f}%)")

        # Display
        cv2.imshow("Video Appliance Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Video detection ended ({frame_count} frames)")


def run_image(image_path):
    """Single image detection"""
    model, classes = load_model()
    transform = get_transform()

    print(f"[INFO] Loading image: {image_path}")
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"[ERROR] Cannot open image: {image_path}")
        return

    # Crop and get coordinates
    crop, (x1, y1, x2, y2) = draw_target_box_and_crop(frame)

    # Predict
    class_idx, confidence = predict(crop, model, transform)
    class_name = classes[class_idx]
    
    if confidence < 0.85 or class_name == "background":
        text = "BACKGROUND"
        color = (150, 150, 150)
    else:
        text = f"{class_name.upper()}: {confidence*100:.1f}%"
        color = COLORS.get(class_name, (255, 255, 255))

    # Draw Green Target Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Add text
    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, color, 2)

    print(f"[DETECTION] {class_name} ({confidence*100:.1f}%)")

    # Display
    cv2.imshow("Image Detection", frame)
    print("[INFO] Press any key to close")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live appliance detection")
    parser.add_argument("--video", type=str, help="Path to video file")
    parser.add_argument("--image", type=str, help="Path to image file")
    args = parser.parse_args()

    if args.image:
        run_image(args.image)
    elif args.video:
        run_video(args.video)
    else:
        run_webcam()
