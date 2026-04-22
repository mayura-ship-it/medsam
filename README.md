# Med3D Project

A comprehensive medical image segmentation project leveraging MedSAM and 3D volume processing.

## Project Structure

- `configs/`: Configuration files for training and inference.
- `MedSAM/`: Core MedSAM implementation (cloned and customized).
- `nnunet/`: Integration with nnU-Net for comparative analysis or preprocessing.
- `notebooks/`: Jupyter notebooks for data exploration and visualization.
- `scripts/`: Python scripts for DICOM processing, preprocessing, and running segmentations.
- `models/`: Folder for model weights (ignored by git due to size).
- `data/`: Data storage (ignored by git).
- `outputs/`: Results and logs.

## Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```
2. Install dependencies (see scripts/preprocess.py for requirements or MedSAM setup).
3. Place model weights in `models/medsam/checkpoints/`.

## Usage

Check the `scripts/` directory for common workflows:
- `dicom_to_nifti.py`: Convert medical images.
- `run_medsam.py`: Execute segmentation.
