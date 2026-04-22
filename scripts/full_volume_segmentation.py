import torch
import numpy as np
import cv2
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

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

# ==============================
# 🔹 LOAD MODEL (ONCE)
# ==============================
model_type = "vit_b"
device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to(device)

predictor = SamPredictor(sam)

print("✅ MedSAM ready")

# ==============================
# 🔹 OUTPUT MASK VOLUME
# ==============================
mask_volume = np.zeros_like(volume, dtype=np.uint8)

# ==============================
# 🔹 PROCESS EACH SLICE
# ==============================
for i in tqdm(range(len(volume))):

    img = volume[i]
    img = (img * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    predictor.set_image(img_rgb)

    h, w, _ = img_rgb.shape
    box = np.array([50, 50, w-50, h-50])

    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True
    )

    # Best mask
    best_idx = np.argmax(scores)
    best_mask = masks[best_idx]

    # ==============================
    # 🔹 CLEAN MASK
    # ==============================
    mask_uint8 = (best_mask * 255).astype(np.uint8)

    kernel = np.ones((5,5), np.uint8)
    clean_mask = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    clean_mask = cv2.medianBlur(clean_mask, 5)
    clean_mask = clean_mask > 0

    # ==============================
    # 🔹 KEEP LARGEST COMPONENT
    # ==============================
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        clean_mask.astype(np.uint8), connectivity=8
    )

    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = (labels == largest_label)
    else:
        final_mask = clean_mask

    # Save mask
    mask_volume[i] = final_mask.astype(np.uint8)

print("✅ Full volume segmentation complete!")

# ==============================
# 🔹 SAVE RESULT
# ==============================
save_path = r"E:\med3d\med3d_project\outputs\masks\medsam\lidc_0001_mask.npy"
np.save(save_path, mask_volume)

print("✅ Mask volume saved at:", save_path)

save_dir = r"E:\med3d\med3d_project\outputs\visualizations"
os.makedirs(save_dir, exist_ok=True)

plt.savefig(f"{save_dir}/slice_{i}.png")