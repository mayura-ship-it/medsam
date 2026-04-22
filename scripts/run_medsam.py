import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor

# ==============================
# 🔹 PATHS
# ==============================
data_path = r"E:\med3d\med3d_project\data\processed\numpy_arrays\lidc_0001.npy"
checkpoint_path = r"E:\med3d\med3d_project\models\medsam\checkpoints\medsam_vit_b.pth"

# ==============================
# 🔹 LOAD DATA
# ==============================
volume = np.load(data_path)
print("✅ Loaded volume:", volume.shape)

# Pick middle slice
slice_idx = len(volume) // 2
img = volume[slice_idx]

# ==============================
# 🔹 PREPARE IMAGE
# ==============================
img = (img * 255).astype(np.uint8)
img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

# ==============================
# 🔹 LOAD MODEL
# ==============================
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)

predictor = SamPredictor(sam)
predictor.set_image(img_rgb)

print("✅ MedSAM model loaded!")

# ==============================
# 🔹 IMPROVED BOX (FULL LUNG REGION)
# ==============================
h, w, _ = img_rgb.shape
box = np.array([50, 50, w-50, h-50])

# ==============================
# 🔹 SEGMENTATION
# ==============================
masks, scores, _ = predictor.predict(
    box=box,
    multimask_output=True
)

print("✅ Segmentation done!")

# ==============================
# 🔹 PICK BEST MASK
# ==============================
best_idx = np.argmax(scores)
best_mask = masks[best_idx]

print("Best score:", scores[best_idx])

# ==============================
# 🔹 CLEAN MASK
# ==============================
mask_uint8 = (best_mask * 255).astype(np.uint8)

kernel = np.ones((5,5), np.uint8)

# Morphological closing
clean_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

# Remove noise
clean_mask = cv2.medianBlur(clean_mask, 5)

# Convert back to boolean
clean_mask = clean_mask > 0

# ==============================
# 🔹 VISUALIZATION
# ==============================

# Original
plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")
plt.title("Original CT Slice")
plt.axis("off")

# Raw mask
plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")
plt.imshow(best_mask, alpha=0.5)
plt.title(f"Raw Mask | Score: {scores[best_idx]:.3f}")
plt.axis("off")

# Clean mask
plt.figure(figsize=(6,6))
plt.imshow(img, cmap="gray")
plt.imshow(clean_mask, alpha=0.5)
plt.title("Cleaned Mask")
plt.axis("off")

plt.show()