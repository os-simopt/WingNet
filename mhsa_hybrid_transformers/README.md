# Hybrid CNN–Transformer Model (MobileNetV4 + MHSA)

This folder provides an auxiliary implementation of a hybrid deep-learning architecture for classifying small aerial objects from video sequences. The model combines a convolutional backbone (MobileNetV4) with a Transformer encoder using Multi-Head Self-Attention (MHSA) to capture temporal dynamics.

The pipeline includes:
- Loading and preprocessing sequence datasets stored as PyTorch `.pt` files
- A hybrid CNN + Transformer model with a frozen timm backbone
- Hyperparameter optimization using Optuna
- Final training of the best-performing configuration
- Evaluation on an independent test set, including a confusion matrix

This code accompanies the research publication, but its optimization/training workflow has not been rerun end-to-end in this release environment.

## Folder Structure

```
project_root/
│
├── mhsa_transformer_train_eval.py     # Full training + optimization + evaluation script
├── ../sample_data/sample_train.pt     # Example training dataset (sequence-level .pt)
├── ../sample_data/sample_test.pt      # Example test dataset
├── splits/                            # Deterministic train/val index files (auto-created)
├── BirdClassification-*.db            # Optuna study database (auto-created)
└── best_mobilenetv4_conv_small.pth    # Final trained model weights (auto-created)
```

## Model Overview

### Feature Extraction
Each frame in a sequence is processed independently using a **MobileNetV4 convolutional backbone** from the `timm` library. The backbone is used in `features_only` mode and kept frozen during training.

For each frame:
- The CNN outputs spatial feature maps.
- Global average pooling reduces the maps to a feature vector.
- All frame vectors form a temporal sequence.

### Temporal Encoding
A **Transformer Encoder** with Multi-Head Self-Attention (MHSA) models temporal relationships across frames:
- `nhead`: number of attention heads
- `num_layers`: number of Transformer blocks
- Feed-forward blocks sized relative to the hidden dimension

### Classification Head
The sequence representation is:
- Normalized
- Dropped out
- Mapped through an MLP
- Classified into 4 classes

## Data Format

The script expects `.pt` files in the format:

```
(seq_ids, list_of_sequences, labels_tensor)
```

Where:
- Each sequence is a tensor of shape `(T, H, W, C)`
- Labels are integer class IDs in `{0, 1, 2, 3}`
- The class `4` is automatically filtered out (ignored)

Padding to a fixed sequence length is handled internally.

## Training and Optimization Pipeline

The script performs:

1. **Dataset loading**
2. **Deterministic train/validation split** using stratified sampling
3. **Optuna optimization**
4. **Final training**
5. **Evaluation on an independent test dataset**

## Usage

### Run Training + Optimization + Evaluation

```
python mhsa_transformer_train_eval.py
```

Outputs:
- `best_mobilenetv4_conv_small.pth`
- `confusion_matrix_hybrid_test.png`
- `BirdClassification-*.db`
- `splits/split_*.json`

## Requirements

Install:

```
pip install torch torchvision timm optuna scikit-learn matplotlib seaborn tqdm
```

## Citation

If using this code in academic work, please cite the associated publication.
