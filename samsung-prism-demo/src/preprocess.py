from torchvision import transforms
from PIL import Image
import torch

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_transform = transforms.Compose([
    transforms.Resize((320, 320)),
    transforms.ToTensor(),                          # [3,320,320], values 0-1
    transforms.Normalize(mean=IMAGENET_MEAN,
                         std=IMAGENET_STD)
])

def load_frame(image_path: str) -> torch.Tensor:
    """
    Load an image from disk and preprocess it.
    Returns: tensor of shape [1, 3, 320, 320]
    """
    img = Image.open(image_path).convert("RGB")
    tensor = _transform(img)        # [3, 320, 320]
    return tensor.unsqueeze(0)      # [1, 3, 320, 320]


# Quick test — run: python src/preprocess.py
if __name__ == "__main__":
    import sys
    path = sys.argv[1]
    t = load_frame(path)
    print("Shape:", t.shape)        # expect [1, 3, 320, 320]
    print("Min/Max:", t.min().item(), t.max().item())