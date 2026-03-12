# Offroad Semantic Segmentation

Semantic segmentation of desert off-road environments using synthetic data from Duality AI's Falcon platform. Built for the **Duality AI Offroad Semantic Scene Segmentation Hackathon**.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

This project trains a **DeepLabV3+** segmentation model with a **ResNet-101** encoder to classify 11 semantic classes in off-road desert scenes. Training runs on Google Colab (T4 GPU) with models saved to Google Drive.

### Segmentation Classes (11)

| ID  | Pixel Value | Class          |                                                                        Color |
| --- | ----------- | -------------- | ---------------------------------------------------------------------------: |
| 0   | 0           | Background     |        ![#000000](https://via.placeholder.com/12/000000/000000?text=+) Black |
| 1   | 100         | Trees          | ![#228B22](https://via.placeholder.com/12/228B22/228B22?text=+) Forest Green |
| 2   | 200         | Lush Bushes    |         ![#00FF00](https://via.placeholder.com/12/00FF00/00FF00?text=+) Lime |
| 3   | 300         | Dry Grass      |          ![#D2B48C](https://via.placeholder.com/12/D2B48C/D2B48C?text=+) Tan |
| 4   | 500         | Dry Bushes     |        ![#8B5A2B](https://via.placeholder.com/12/8B5A2B/8B5A2B?text=+) Brown |
| 5   | 550         | Ground Clutter |        ![#808000](https://via.placeholder.com/12/808000/808000?text=+) Olive |
| 6   | 600         | Flowers        |      ![#FF00FF](https://via.placeholder.com/12/FF00FF/FF00FF?text=+) Magenta |
| 7   | 700         | Logs           | ![#8B4513](https://via.placeholder.com/12/8B4513/8B4513?text=+) Saddle Brown |
| 8   | 800         | Rocks          |         ![#808080](https://via.placeholder.com/12/808080/808080?text=+) Gray |
| 9   | 7100        | Landscape      |       ![#A0522D](https://via.placeholder.com/12/A0522D/A0522D?text=+) Sienna |
| 10  | 10000       | Sky            |     ![#87CEEB](https://via.placeholder.com/12/87CEEB/87CEEB?text=+) Sky Blue |

## Repository Structure

```
├── train.py                  # Google Colab training notebook (exported as .py)
├── test.py                   # Google Colab testing/inference notebook (exported as .py)
├── Offroad_Segmentation_Scripts/
│   ├── train_winning.py      # Standalone training script (DeepLabV3+ / smp)
│   ├── test_winning.py       # Standalone test script with full evaluation
│   ├── train_segmentation.py # Baseline DINOv2 + segmentation head training
│   ├── test_segmentation.py  # Baseline DINOv2 test script
│   ├── visualize.py          # Mask colorization utility
│   ├── generate_report.py    # Auto-generates REPORT.md from results
│   ├── config.json           # Model configuration (auto-generated during training)
│   ├── best_model.pth        # Best model weights (saved during training)
│   ├── README.md             # Detailed script-level documentation
│   ├── REPORT.md             # Generated hackathon report
│   ├── results/              # Training curves, metrics, logs
│   └── test_results/         # Test predictions, confusion matrix, failure analysis
├── Offroad_Segmentation_Training_Dataset/
│   └── train/                # 2857 training images (960×540)
│       ├── Color_Images/
│       └── Segmentation/
│   └── val/                  # 317 validation images
│       ├── Color_Images/
│       └── Segmentation/
└── Offroad_Segmentation_testImages/
    ├── Color_Images/         # 1002 test RGB images
    └── Segmentation/         # 1002 test ground truth masks
```

## Quick Start

### Option 1: Google Colab (Recommended)

**Training:**

1. Upload the dataset to Google Drive under `hackathon/`
2. Open `train.py` in Google Colab (or use the [original notebook link](https://colab.research.google.com/drive/1AoCpqNnFGSiVIpXLTmk8HYAxJsOvL6TX))
3. Set runtime to **T4 GPU** (`Runtime → Change runtime type → T4 GPU`)
4. Run all cells — the model auto-saves `best_model.pth` to Google Drive

**Testing/Inference:**

1. Ensure `best_model.pth` and `config.json` are in your Drive at `hackathon/OffroadSegmentation/`
2. Upload test images to `hackathon/Offroad_Segmentation_testImages/`
3. Open `test.py` in Colab (or use the [original notebook link](https://colab.research.google.com/drive/1lw5LOUPuet8A3tiIoz0nH_blCRGswMKm))
4. Run all cells — predictions are saved to `hackathon/OffroadSegmentation/predictions/`

**Test Dataset:** Available at [Google Drive](https://drive.google.com/drive/folders/1VItRj3wgJelbRJ1jIwiZOVs8VvdLvfld)

### Option 2: Local (CPU/GPU)

```bash
# 1. Create conda environment
conda create -n hackathon python=3.10 -y
conda activate hackathon

# 2. Install PyTorch
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y  # CPU
# OR for GPU: conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# 3. Install dependencies
pip install segmentation-models-pytorch albumentations opencv-python matplotlib tqdm scikit-learn

# 4. Train
cd Offroad_Segmentation_Scripts
python train_winning.py --arch DeepLabV3Plus --encoder resnet50 --epochs 40 --batch_size 4 --img_size 512

# 5. Test
python test_winning.py --model_path best_model.pth --num_comparisons 20
```

## Training Configuration

| Parameter     | Colab Default       | Description                                      |
| ------------- | ------------------- | ------------------------------------------------ |
| Architecture  | DeepLabV3+          | Also supports UnetPlusPlus, FPN, PSPNet          |
| Encoder       | ResNet-101          | ImageNet pretrained; also try resnet34, resnet50 |
| Image Size    | 320×320             | Higher = better quality, more memory             |
| Batch Size    | 16                  | Reduce if OOM (2–4 for CPU)                      |
| Epochs        | 30                  | Early stopping with patience=7                   |
| Learning Rate | 1e-3                | Cosine annealing to 1e-6                         |
| Optimizer     | AdamW               | Weight decay 1e-4                                |
| Loss          | CE + Dice (0.5/0.5) | Weighted CrossEntropy + Dice loss                |

### Class Weights

Manually tuned to handle severe class imbalance:

```
Background:     0.0  (absent in training)
Trees:          8.0  (rare)
Lush Bushes:    6.0  (very rare)
Dry Grass:      1.5  (common)
Dry Bushes:     3.0  (uncommon)
Ground Clutter: 8.0  (rare)
Flowers:        5.0  (absent)
Logs:           5.0  (absent)
Rocks:          3.0  (uncommon)
Landscape:      0.8  (common)
Sky:            0.3  (dominant)
```

## Methodology

### Architecture

**DeepLabV3+** with atrous spatial pyramid pooling (ASPP) for multi-scale feature extraction. The ResNet encoder is pretrained on ImageNet for strong transfer learning.

### Data Augmentation (Albumentations)

- Horizontal flip, random brightness/contrast
- Hue/saturation/value shifts
- Random fog and sun flare
- Gaussian blur
- Random resized crop (scale 0.5–1.0)

### Loss Function

Combined **CrossEntropy + Dice Loss** (equal weight):

- CrossEntropy handles pixel-level classification with class weighting
- Dice loss directly optimizes region overlap, robust to class imbalance

### Test-Time Augmentation (TTA)

During inference, predictions from the original image and its horizontal flip are averaged for improved accuracy.

### Training Strategy

- **Cosine annealing** LR schedule for smooth convergence
- **Early stopping** (patience=7) to prevent overfitting
- **Gradient clipping** (max norm=1.0) for stable training
- Auto-saves best model by validation mIoU

## Outputs

### Training Outputs (saved to Google Drive)

| File                         | Description                          |
| ---------------------------- | ------------------------------------ |
| `best_model.pth`             | Best model checkpoint (by val mIoU)  |
| `config.json`                | Architecture and class configuration |
| `training_history.json`      | Full training metrics per epoch      |
| `training_curves.png`        | Loss, mIoU, accuracy, LR curves      |
| `per_class_iou.png`          | Per-class IoU bar chart              |
| `confusion_matrix.png`       | Normalized confusion matrix          |
| `validation_predictions.png` | GT vs prediction visual comparisons  |

### Test Outputs

| File/Folder            | Description                             |
| ---------------------- | --------------------------------------- |
| `predictions/`         | Predicted masks (original pixel values) |
| `masks/`               | Raw prediction masks (class IDs 0–10)   |
| `masks_color/`         | Colored prediction visualizations       |
| `comparisons/`         | Side-by-side GT vs prediction images    |
| `failure_cases/`       | Worst predictions with error maps       |
| `confusion_matrix.png` | Test confusion matrix                   |
| `test_metrics.txt`     | Per-class IoU and failure analysis      |

## Key Challenges

| Challenge                                                               | Solution                               |
| ----------------------------------------------------------------------- | -------------------------------------- |
| **Severe class imbalance** (Landscape ~35%, Lush Bushes <1%)            | Manual class weights + Dice loss       |
| **Domain shift** between train/test environments                        | Aggressive augmentation pipeline       |
| **Missing classes** (Background, Flowers, Logs have ~0 training pixels) | Zero-weight for absent classes         |
| **CPU-only local training**                                             | Efficient encoder + Colab GPU workflow |

## Requirements

- Python 3.10
- PyTorch ≥ 2.0
- segmentation-models-pytorch
- albumentations
- opencv-python
- matplotlib
- tqdm
- scikit-learn (for confusion matrix)
- seaborn (for heatmaps, Colab only)
