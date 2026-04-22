import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Path to your dataset
dicom_path = r"E:\med3d\med3d_project\data\raw\dicom\LIDC-IDRI-0001"

# 🔹 Read DICOM series
reader = sitk.ImageSeriesReader()

dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
reader.SetFileNames(dicom_names)

image = reader.Execute()

# 🔹 Convert to numpy array
array = sitk.GetArrayFromImage(image)

print("✅ CT Scan Loaded Successfully!")
print("Shape:", array.shape)

# 🔹 Show few slices
for i in range(0, len(array), 20):
    plt.imshow(array[i], cmap="gray")
    plt.title(f"Slice {i}")
    plt.axis("off")
    plt.show()

# Show middle slices (lungs usually here)
mid = len(array) // 2

for i in range(mid-10, mid+10):
    plt.imshow(array[i], cmap="gray")
    plt.title(f"Slice {i}")
    plt.axis("off")
    plt.show()

# 🔹 Normalize (0–1)
array = (array - np.min(array)) / (np.max(array) - np.min(array))

print("Normalized Range:", np.min(array), np.max(array))