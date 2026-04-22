import SimpleITK as sitk

dicom_path = r"E:\med3d\med3d_project\data\raw\dicom\LIDC-IDRI-0001"
output_path = r"E:\med3d\med3d_project\data\processed\lidc_0001.nii.gz"

reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
reader.SetFileNames(dicom_names)

image = reader.Execute()

sitk.WriteImage(image, output_path)

print("✅ NIfTI saved:", output_path)