# xLSTM Hybrid Model Training and Evaluation

This repository provides the full training, retraining, and evaluation workflow for the **MobileNetV4 + xLSTM hybrid architecture** used for avian species classification from video/image sequences.

This auxiliary experiment script has not been rerun end-to-end as part of this repository release.

---

## 1. Overview

The hybrid classification architecture integrates:

- **MobileNetV4 Conv-Small**
  (frozen backbone for per-frame feature extraction)

- **xLSTM (Meta AI)**
  (sequence modeling with mLSTM + sLSTM blocks)

- **MLP Classification Head**
  (sequence-level logits for 4 classes: Kite, Bird, Aircraft, Other)

This pipeline includes:

- Deterministic and reusable train/validation splits
- Optuna hyperparameter optimization
- Optional retraining without optimization
- Strict separation of TRAIN and TEST sets
- Automatic checkpointing
- Confusion matrix generation
- Fully configurable command-line interface

---

## 2. Script Features

### 2.1 Reproducible Train/Val Split

A stratified split is generated **only from the TRAIN dataset** and saved under:

```
splits/split_<signature>.json
```

Any script using the same TRAIN file, seed, and ratio will reuse the exact same split.

---

### 2.2 Three Operational Modes

#### **(A) Full workflow with Optuna (default)**
```
python xlstm_train_eval.py
```

Steps:

1. Load TRAIN and TEST
2. Create or load deterministic split
3. Run Optuna (100 trials)
4. Train final model using best params
5. Evaluate on TEST

---

#### **(B) Retraining without Optuna**
```
python xlstm_train_eval.py --retrain_best
```

Loads parameters from the Optuna study:

```
BirdClassification-mobilenetv4_conv_small-xlstm.db
```

---

#### **(C) Train with explicit hyperparameters**
```
python xlstm_train_eval.py --params_json '{"lr":3e-4,"optimizer":"AdamW",...}'
```

Use this mode to bypass Optuna completely.

---

## 3. Required Dataset Format

The script expects `.pt` dataset files with the structure:

```
(seq_ids, sequences, labels)
```

- `sequences[i]` → tensor `(T, H, W, C)`
- `labels[i]` → integer in `{0, 1, 2, 3, 4}`
- Label `4` is excluded automatically (unlabeled)

Both files must be preprocessed beforehand:

```
sample_data/sample_train.pt
sample_data/sample_test.pt
```

---

## 4. Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--retrain_best` | False | Use existing Optuna params |
| `--params_json` | None | Skip Optuna, use explicit params |
| `--train` | `sample_data/sample_train.pt` | Train dataset (repository-relative default) |
| `--test` | `sample_data/sample_test.pt` | Test dataset (repository-relative default) |
| `--val_ratio` | 0.15 | Validation fraction |
| `--batch_size` | 16 | Batch size |
| `--max_seq_len` | 32 | Sequence truncation/padding |
| `--num_workers` | 2 | DataLoader workers |
| `--pin_memory` | False | Pin host memory for faster transfers |
| `--seed` | 42 | Reproducibility |

---

## 5. Output Files

| File | Purpose |
|------|----------|
| `best_mobilenetv4_conv_small_xlstm.pth` | Best checkpoint |
| `BirdClassification-mobilenetv4_conv_small-xlstm.db` | Optuna study |
| `confusion_matrix_xLSTM_TEST.png` | Test confusion matrix |

---

## 6. CUDA Requirements for xLSTM

The xLSTM backend compiles C++/CUDA kernels at runtime.

This requires **full CUDA Toolkit**, including headers such as:

- `cuda_runtime.h`
- `cuda_runtime_api.h`

If you see errors like:

```
fatal error: cuda_runtime.h: No such file or directory
```

then install CUDA Toolkit:

### Debian/Ubuntu
```
sudo apt install cuda-toolkit-12-8
```

### Module load system
```
module load cuda/12.8
```

### Verify installation
```
which nvcc
nvcc --version
```

---

## 7. Running the Pipeline

### Standard workflow:
```
python xlstm_train_eval.py
```

### Retrain best configuration:
```
python xlstm_train_eval.py --retrain_best
```

### Manual hyperparameters:
```
python xlstm_train_eval.py --params_json FILE.json
```

---

## 8. Citation

If using this code in academic work, please cite the associated publication.
