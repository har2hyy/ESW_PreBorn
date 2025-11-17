import cv2
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

# Select device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (Base = ViT-B)
model = DepthAnythingV2(encoder="vitb", features=128, out_channels=[96, 192, 384, 768])
model.load_state_dict(torch.load("checkpoints/depth_anything_v2_vitb.pth", map_location=DEVICE))
model = model.to(DEVICE).eval()

# Read input image
img = cv2.imread("sample.jpg")  # replace with your image path
if img is None:
    raise FileNotFoundError("Image not found! Please put an image as 'sample.jpg' in this folder.")

# Inference
with torch.no_grad():
    depth = model.infer_image(img)

# ---- BRIGHTNESS / CONTRAST ADJUSTMENT ----
# Step 1: Scale depth to exaggerate near–far differences
depth_scaled = depth * 10.0

# Step 2: Normalize depth map to [0,1] for safe manipulation
depth_scaled = (depth_scaled - depth_scaled.min()) / (depth_scaled.max() - depth_scaled.min())

# Step 3: Brightness + contrast adjustment
# alpha > 1.0 increases contrast, beta > 0 increases brightness
alpha = 1.2   # contrast control (1.0 = no change)
beta = 0.1    # brightness control (0 = no change)
depth_adjusted = np.clip(alpha * depth_scaled + beta, 0, 1)

# Step 4: Convert to uint8 image
depth_norm = (depth_adjusted * 255).astype("uint8")

# Save the brightened grayscale image
cv2.imwrite("depth_output_bright.png", depth_norm)

print("✅ Brightened depth map saved as depth_output_bright.png")
