import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# 🔹 Load DICOM
dicom_path = r"E:\med3d\med3d_project\data\raw\dicom\LIDC-IDRI-0001"

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
reader.SetFileNames(dicom_names)

image = reader.Execute()
array = sitk.GetArrayFromImage(image)

print("Original shape:", array.shape)

# 🔹 Convert to Hounsfield Units (HU)
# Rescale using metadata
intercept = image.GetMetaData("0028|1052") if image.HasMetaDataKey("0028|1052") else "0"
slope = image.GetMetaData("0028|1053") if image.HasMetaDataKey("0028|1053") else "1"

intercept = float(intercept)
slope = float(slope)

hu_array = array * slope + intercept

# 🔹 Lung windowing
# Typical lung range: -1000 to 400 HU
min_hu = -1000
max_hu = 1000

hu_array = np.clip(hu_array, min_hu, max_hu)

# Normalize again (0–1)
hu_array = (hu_array - min_hu) / (max_hu - min_hu)

print("After preprocessing:", hu_array.shape)

# 🔹 Show slices
mid = len(hu_array) // 2

for i in range(mid-5, mid+5):
    plt.imshow(hu_array[i], cmap="gray")
    plt.title(f"Preprocessed Slice {i}")
    plt.axis("off")
    plt.show()


# 🔹 Save processed array
import os

save_dir = r"E:\med3d\med3d_project\data\processed\numpy_arrays"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "lidc_0001.npy")

np.save(save_path, hu_array)

print("✅ Processed data saved at:", save_path)



save_img_dir = r"E:\med3d\med3d_project\data\processed\images"
os.makedirs(save_img_dir, exist_ok=True)

for i in range(len(hu_array)):
    plt.imsave(f"{save_img_dir}/slice_{i}.png", hu_array[i], cmap="gray")