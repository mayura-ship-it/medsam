import numpy as np
import matplotlib.pyplot as plt
import os

# ==============================
# 🔹 PATHS
# ==============================
volume_path = r"E:\med3d\med3d_project\data\processed\numpy_arrays\lidc_0001.npy"
mask_path = r"E:\med3d\med3d_project\outputs\masks\medsam\lidc_0001_mask.npy"

save_dir = r"E:\med3d\med3d_project\outputs\visualizations"
os.makedirs(save_dir, exist_ok=True)

# ==============================
# 🔹 LOAD DATA
# ==============================
volume = np.load(volume_path)
mask_volume = np.load(mask_path)

print("✅ Loaded volume:", volume.shape)
print("✅ Loaded mask:", mask_volume.shape)

# ==============================
# 🔹 VISUALIZATION (LUNG SLICES ONLY)
# ==============================
print("📊 Showing lung slices only...")

# Lung region approx: slice 40 → 100
for i in range(55, 95, 10):

    img = volume[i]
    mask = mask_volume[i]

    plt.figure(figsize=(12,4))

    # Original
    plt.subplot(1,3,1)
    plt.imshow(img, cmap="gray")
    plt.title(f"Original Slice {i}")
    plt.axis("off")

    # Mask
    plt.subplot(1,3,2)
    plt.imshow(mask, cmap="gray")
    plt.title("Mask")
    plt.axis("off")

    # Overlay (Final polished result)
    plt.subplot(1,3,3)
    plt.imshow(img, cmap="gray")
    plt.imshow(mask, alpha=0.5)
    plt.title("Final Overlay (Polished)")
    plt.axis("off")

    # Save image
    save_path = os.path.join(save_dir, f"lung_slice_{i}.png")
    plt.savefig(save_path)

    print(f"✅ Saved: {save_path}")

    plt.show()

print("🎉 Lung visualization complete!")