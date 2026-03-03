# AI600 Deep Learning — PA2: Quick, Draw! Challenge

**Lahore University of Management Sciences**
School of Science and Engineering — Department of Electrical Engineering
Spring 2026

---

## Overview

This repository contains the implementation for Programming Assignment 2 of AI600 Deep Learning. The goal is to classify 28×28 hand-drawn doodles from the [Quick, Draw!](https://quickdraw.withgoogle.com/data) dataset into 15 categories using only Multi-Layer Perceptrons (MLPs) — no Convolutional Neural Networks allowed.

Three models are implemented and compared:
- **Pancake** — wide and shallow (2 hidden layers, 1024 neurons)
- **Tower** — deep and narrow (8 hidden layers, 128 neurons)
- **Champion** — optimized for accuracy using patch embeddings, augmentation, and Mixup

---

## Results

| Model    | Parameters  | Epochs | Best Val Accuracy |
|----------|-------------|--------|-------------------|
| Pancake  | 1,868,815   | 20     | 76.48%            |
| Tower    | 220,047     | 25     | 73.98%            |
| Champion | 570,143     | 35     | **85.52%**        |

---

## Dataset

The dataset contains 60,000 training images across 15 classes:
`apple`, `baseballbat`, `basketball`, `clock`, `compass`, `cookie`, `donut`, `ladder`, `mountain`, `pizza`, `rabbit`, `soccerball`, `spider`, `t-shirt`, `wheel`

Each image is a 28×28 grayscale bitmap flattened to 784 values, normalized to [0, 1].

Download the processed `.npz` files and place them in the project root:
```
quickdraw_train.npz
quickdraw_test.npz
```

---

## Repository Structure

```
.
├── PA2.ipynb                  # Main notebook (Pancake, Tower, Champion)
├── quickdraw_train.npz        # Training data (not included, download separately)
├── quickdraw_test.npz         # Test data (not included, download separately)
├── pancake_weights.pth        # Saved Pancake model weights
├── tower_weights.pth          # Saved Tower model weights
├── champion_weights.pth       # Saved Champion model weights
├── submission.txt             # Leaderboard predictions (comma-separated)
├── confusion_matrix.png       # Champion confusion matrix on validation set
├── Pancake_curves.png         # Pancake training curves
├── Tower_curves.png           # Tower training curves
├── Champion_curves.png        # Champion training curves
└── README.md
```

---

## Requirements

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn pandas
```

Tested with Python 3.10, PyTorch 2.x.

---

## How to Run

1. Clone the repository and install dependencies
2. Place `quickdraw_train.npz` and `quickdraw_test.npz` in the project root
3. Open `PA2.ipynb` and run all cells top to bottom
4. Predictions will be saved to `submission.txt`

Training automatically uses CUDA if available, then MPS (Apple Silicon), then CPU.

---

## Champion Model Architecture

The Champion model improves on the Pancake and Tower baselines with three key additions:

**1. Patch Embeddings**
The 28×28 image is split into 49 non-overlapping 4×4 patches. Each patch is projected to a 16-dimensional embedding via a shared linear layer, preserving local spatial structure without using convolutions.

**2. Data Augmentation** (training only)
- Random horizontal flip (50%)
- Random rotation (±15°)
- Random affine translation (±10%)
- Random erasing (30% probability)

**3. Mixup Regularization**
Training image pairs are blended with soft labels (α = 0.3), forcing the model to learn smooth class boundaries.

**Architecture:**
```
PatchEmbed(784) → FC(512) → BN → GELU → Dropout(0.30)
               → FC(256) → BN → GELU → Dropout(0.25)
               → FC(128) → BN → GELU → Dropout(0.20)
               → FC(15)
```

**Training:** AdamW (lr=0.01, wd=0.0003) + OneCycleLR (max_lr=0.05) + gradient clipping

---

## Inference

To generate predictions on the test set using the saved Champion weights:

```python
import torch
from PA2 import ChampionMLP, QuickDrawDataset
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = ChampionMLP().to(DEVICE)
model.load_state_dict(torch.load('champion_weights.pth', map_location=DEVICE))
model.eval()

test_ds = QuickDrawDataset('quickdraw_test.npz', mode='test')
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

preds = []
with torch.no_grad():
    for batch in test_loader:
        preds.extend(model(batch.to(DEVICE)).argmax(1).cpu().numpy())

print(','.join(map(str, preds)))
```

---

## Author

Roll Number: 24-2528-0035
Course: AI600 Deep Learning — Spring 2026
Institution: LUMS, Department of Electrical Engineering
