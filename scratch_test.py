import cv2
import torch
from torchvision import transforms
from PIL import Image
from test_model import model, DEVICE, tf as test_tf
from live_detection import load_model, get_transform, predict

img_path = "dataset/test/air_conditioner/11_jpg.rf.0BY6QkXXG55SVhK9B35C.jpg"

# 1. PIL way (test_model.py approach)
img_pil = Image.open(img_path).convert("RGB")
tensor_pil = test_tf(img_pil).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    res_pil = torch.softmax(model(tensor_pil), dim=1)

# 2. CV2 way (live_detection.py approach)
img_cv2 = cv2.imread(img_path)
model_live, classes = load_model()
frame_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
transform_live = get_transform()
tensor_cv2 = transform_live(frame_rgb).unsqueeze(0).to(DEVICE)
with torch.no_grad():
    res_cv2 = torch.softmax(model_live(tensor_cv2), dim=1)

print(f"PIL result: {res_pil.cpu().numpy()}")
print(f"CV2 result: {res_cv2.cpu().numpy()}")
