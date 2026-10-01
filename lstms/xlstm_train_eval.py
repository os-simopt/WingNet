# -*- coding: utf-8 -*-
"""
PyTorch script for training, retraining, and evaluating a hybrid
(MobileNet backbone + xLSTM) model for avian species classification
from motion sequences.

The script:
  - loads train and test sequence datasets from .pt files,
  - constructs a deterministic, stratified train/validation split from the train set,
  - optionally runs Optuna hyperparameter optimization on the train/val split,
  - trains the final model with the selected hyperparameters, and
  - evaluates on the held-out test set, saving a confusion matrix.

Expected dataset format (.pt files):
    torch.save((seq_ids, seqs_list, labels_tensor), path)
    - seq_ids: 1D tensor or list of sequence identifiers (unused here)
    - seqs_list: list of tensors with shape (T, H, W, C)
    - labels_tensor: 1D tensor of integer labels in {0,1,2,3,4}
                     (label 4 is ignored in this script)

Usage examples:
  # Full workflow: Optuna search → train final model → test evaluation
  python xlstm_train_eval.py

  # Retrain only, using best Optuna trial stored in the study database
  python xlstm_train_eval.py --retrain_best

  # Skip Optuna entirely, provide hyperparameters explicitly as JSON
  python xlstm_train_eval.py \
      --params_json '{"transformer_heads":8,"transformer_layers":3,"mlp_units":512,"dropout_rate":0.2,"lr":3e-4,"optimizer":"AdamW","weight_decay":1e-4}'

  # Use a different backbone (e.g. MobileNetV2 or EfficientNetV2-B1)
  python xlstm_train_eval.py --backbone mobilenetv2_120d
"""

import os
import json
import argparse
import hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

import timm
import optuna
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

# --- 1. Global configuration -------------------------------------------------

print("PyTorch Version:", torch.__version__)
print("Optuna Version:", optuna.__version__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Default backbone; can be overridden via CLI.
TIMM_MODEL_NAME = "mobilenetv4_conv_small"

# Fixed sequence model identifier (used in filenames and study name).
SEQ_MODEL_NAME = "xlstm"

# Global Optuna study name; updated in main() once backbone is known.
study_name = "BirdClassification-" + TIMM_MODEL_NAME + "-" + SEQ_MODEL_NAME


# --- 1a. Reproducibility -----------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """Set all relevant RNG seeds for deterministic behavior."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


# --- 1b. Split persistence (reusable across scripts) -------------------------


def _split_signature(train_pt_path: str, val_ratio: float, seed: int) -> str:
    """Create a stable signature for the train/val split based on data path, ratio, and seed."""
    h = hashlib.sha256()
    h.update(os.path.abspath(train_pt_path).encode("utf-8"))
    h.update(str(val_ratio).encode("utf-8"))
    h.update(str(seed).encode("utf-8"))
    return h.hexdigest()[:16]


def get_or_create_split(
    labels: np.ndarray,
    *,
    train_pt_path: str,
    val_ratio: float,
    seed: int,
    split_dir: str = "splits",
):
    """
    Return (train_idx, val_idx) as NumPy arrays.

    If a split file already exists for the given (train_pt_path, val_ratio, seed),
    it is loaded. Otherwise, a new stratified split is created and stored.
    """
    os.makedirs(split_dir, exist_ok=True)
    sig = _split_signature(train_pt_path, val_ratio, seed)
    split_path = os.path.join(split_dir, f"split_{sig}.json")

    if os.path.exists(split_path):
        with open(split_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return np.array(data["train_idx"], dtype=int), np.array(data["val_idx"], dtype=int)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    all_idx = np.arange(len(labels))
    (train_idx, val_idx), = sss.split(all_idx, labels)

    with open(split_path, "w", encoding="utf-8") as f:
        json.dump({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}, f)

    return train_idx, val_idx


# --- 2. Dataset --------------------------------------------------------------


class BirdSequenceDataset(Dataset):
    """
    Dataset wrapper for sequence-level .pt files produced by the BirdRecorder pipeline.

    Input file format:
        torch.save((seq_ids, seqs_list, labels_tensor), path)
    where:
        seqs_list[i] has shape (T, H, W, C)
        labels_tensor[i] is an integer class label in {0,1,2,3,4}
    Class 4 is discarded in this implementation.
    """

    def __init__(self, pt_file_path: str, max_seq_len: int = 32) -> None:
        if not os.path.exists(pt_file_path):
            raise FileNotFoundError(f"Dataset file not found at {pt_file_path}")

        print(f"Loading data from {pt_file_path}...")
        # seq_ids are unused here
        _, seqs_list, labels_tensor = torch.load(
            pt_file_path, map_location="cpu", weights_only=True
        )

        labels = labels_tensor.numpy()
        # Discard label 4
        valid_indices = np.where(labels != 4)[0]

        self.labels = torch.from_numpy(labels[valid_indices]).long()
        sequences = [seqs_list[i] for i in valid_indices]
        self.padded_sequences = self._pad_and_normalize(sequences, max_seq_len)

        print(f"Found {len(self.labels)} valid sequences (labels != 4).")

    @staticmethod
    def _pad_and_normalize(sequences, max_len: int) -> torch.Tensor:
        """
        Pad all sequences to max_len and normalize into [0, 1].

        Each sequence tensor has shape (T, H, W, C).
        After padding and rearrangement, stored shape is (N, T, H, W, C).
        """
        padded = []
        for seq_tensor in sequences:
            # T, H, W, C -> C, T, H, W
            seq_tensor = seq_tensor.permute(3, 0, 1, 2)
            seq_len = seq_tensor.shape[1]

            if seq_len >= max_len:
                padded_seq = seq_tensor[:, :max_len, :, :]
            else:
                padding = torch.zeros(
                    seq_tensor.shape[0],
                    max_len - seq_len,
                    seq_tensor.shape[2],
                    seq_tensor.shape[3],
                    dtype=seq_tensor.dtype,
                )
                padded_seq = torch.cat([seq_tensor, padding], dim=1)

            # C, T, H, W -> T, H, W, C
            padded.append(padded_seq.permute(1, 2, 3, 0))

        stacked = torch.stack(padded)  # (N, T, H, W, C)
        return stacked

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        # (T, H, W, C) -> (T, C, H, W), normalize to [0, 1]
        sequence = self.padded_sequences[idx].permute(0, 3, 1, 2).float() / 255.0
        label = self.labels[idx]
        return sequence, label


# --- 3. Model architecture ---------------------------------------------------


class FeatureExtractor(nn.Module):
    """
    Per-frame feature extractor based on a timm backbone (features_only=True).
    Applies global average pooling to obtain one embedding per frame.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t = x.size(0), x.size(1)
        x_reshaped = x.contiguous().view(b * t, *x.size()[2:])  # (B*T, C, H, W)

        # timm features_only returns a list; use the last stage
        feats = self.backbone(x_reshaped)[0]  # (B*T, C_feat, H_feat, W_feat)

        pooled = self.pooling(feats)  # (B*T, C_feat, 1, 1)
        pooled = pooled.view(b, t, -1)  # (B, T, C_feat)
        return pooled


class HybridModel(nn.Module):
    """
    Hybrid architecture: CNN backbone (MobileNet / EfficientNet) + xLSTM stack + MLP classifier.
    """

    def __init__(self, hp: dict) -> None:
        super().__init__()

        transformer_heads = hp["transformer_heads"]
        transformer_layers = hp["transformer_layers"]
        mlp_units = hp["mlp_units"]
        dropout_rate = hp["dropout_rate"]

        # CNN backbone
        backbone = timm.create_model(
            TIMM_MODEL_NAME,
            pretrained=True,
            features_only=True,
            out_indices=[4],
        )
        for p in backbone.parameters():
            p.requires_grad = False

        self.feature_extractor = FeatureExtractor(backbone)

        # Feature dimension of the last backbone stage (MobileNetV4 conv small uses 960)
        feature_dim = 960

        # xLSTM configuration
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=transformer_heads,
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda" if torch.cuda.is_available() else "cpu",
                    num_heads=transformer_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn="gelu",
                ),
            ),
            context_length=32,
            num_blocks=transformer_layers,
            embedding_dim=feature_dim,
            slstm_at=[1] if transformer_layers == 3 else [],
        )
        self.xlstm_stack = xLSTMBlockStack(cfg)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, mlp_units),
            nn.ReLU(),
            nn.Linear(mlp_units, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        x: (B, T, C, H, W)
        """
        x = self.feature_extractor(x)  # (B, T, D)
        x = self.xlstm_stack(x)        # (B, T, D)
        x = x.mean(dim=1)              # (B, D)
        logits = self.classifier(x)    # (B, 4)
        return logits


# --- 4. Optuna optimization --------------------------------------------------


def objective(trial: optuna.Trial, train_loader: DataLoader, val_loader: DataLoader) -> float:
    """Optuna objective: maximize macro F1 on the validation split."""
    params = {
        "transformer_heads": trial.suggest_categorical("transformer_heads", [4, 8]),
        "transformer_layers": trial.suggest_int("transformer_layers", 1, 3),
        "mlp_units": trial.suggest_categorical("mlp_units", [256, 512]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
    }

    model = HybridModel(params).to(DEVICE)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])

    if optimizer_name == "AdamW":
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    best_f1 = 0.0
    epochs_without_improvement = 0
    patience = 5

    for epoch in range(15):
        model.train()
        for sequences, labels in train_loader:
            sequences = sequences.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(sequences)
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        trial.report(f1, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if f1 > best_f1:
            best_f1 = f1
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    return best_f1


def run_optimization(train_loader: DataLoader, val_loader: DataLoader) -> dict:
    """Run Optuna optimization and return the best hyperparameters."""
    storage_name = f"sqlite:///{study_name}.db"

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )

    print("\nStarting optimization for xLSTM hybrid model (100 trials).")
    study.optimize(lambda trial: objective(trial, train_loader, val_loader), n_trials=100)
    print("\nOptimization finished.")
    print(f"  Best F1: {study.best_value}")
    print("  Best parameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study.best_params


# --- 5. Training and evaluation ---------------------------------------------


def _load_state_dict(path: str) -> dict:
    """
    Load a state dict from disk, using weights_only=True when supported by the
    installed PyTorch version.
    """
    if hasattr(torch.load, "__code__") and "weights_only" in torch.load.__code__.co_varnames:  # type: ignore[attr-defined]
        state = torch.load(path, map_location=DEVICE, weights_only=True)
    else:
        state = torch.load(path, map_location=DEVICE)
    return state


def train_final_model(best_params: dict, train_loader: DataLoader, val_loader: DataLoader) -> HybridModel:
    """Train the final xLSTM model using the provided hyperparameters."""
    print("\n--- Training final xLSTM model ---")

    model = HybridModel(best_params).to(DEVICE)

    lr = best_params["lr"]
    if best_params["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=best_params["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    best_val_f1 = 0.0
    epochs_without_improvement = 0
    patience = 10

    ckpt_path = f"best_{TIMM_MODEL_NAME}_{SEQ_MODEL_NAME}.pth"

    for epoch in range(50):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/50 [Train]")
        for sequences, labels in pbar:
            sequences = sequences.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item())

        model.eval()
        all_preds, all_labels = [], []
        val_loss = 0.0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch + 1}/50 [Val]")
            for sequences, labels in pbar_val:
                sequences = sequences.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)

                outputs = model(sequences)
                val_loss += criterion(outputs, labels).item()
                preds = torch.argmax(outputs, dim=1)

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        mean_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}: Val Loss: {mean_val_loss:.4f}, Val F1: {f1:.4f}")

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> New best model saved to {ckpt_path} with F1 = {best_val_f1:.4f}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

    state = _load_state_dict(ckpt_path)
    model.load_state_dict(state)
    return model


def evaluate_final_model(test_loader: DataLoader) -> None:
    """Evaluate the best saved model on the test set and save a confusion matrix plot."""
    print("\n--- Final model evaluation on TEST set ---")
    class_names = ["Kite", "Bird", "Aircraft", "Other"]

    model_path = f"best_{TIMM_MODEL_NAME}_{SEQ_MODEL_NAME}.pth"
    if not os.path.exists(model_path):
        raise RuntimeError("Missing trained weights. Train the model before evaluation.")

    storage_name = f"sqlite:///{study_name}.db"
    db_path = storage_name.replace("sqlite:///", "")

    if os.path.exists(db_path):
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_params = study.best_params
    else:
        # Fallback: manually specified default hyperparameters
        best_params = {
            "transformer_heads": 8,
            "transformer_layers": 3,
            "mlp_units": 512,
            "dropout_rate": 0.2,
            "lr": 3e-4,
            "optimizer": "AdamW",
            "weight_decay": 1e-4,
        }

    model = HybridModel(best_params).to(DEVICE)

    print(f"Loading saved weights from {model_path}")
    state = _load_state_dict(model_path)
    model.load_state_dict(state)

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(sequences)
            preds = torch.argmax(outputs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix for xLSTM Hybrid Model (Test)")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{TIMM_MODEL_NAME}_{SEQ_MODEL_NAME}_TEST.png")
    plt.close()
    print(f"Confusion matrix saved to confusion_matrix_{TIMM_MODEL_NAME}_{SEQ_MODEL_NAME}_TEST.png")


# --- 6. Orchestration --------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train, retrain, and evaluate a MobileNet + xLSTM hybrid model for bird species classification."
    )
    parser.add_argument(
        "--retrain_best",
        action="store_true",
        help="Skip Optuna, load existing study best hyperparameters, and retrain.",
    )
    parser.add_argument(
        "--params_json",
        type=str,
        default=None,
        help="Bypass Optuna entirely: JSON string with full hyperparameters including lr/optimizer/weight_decay.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation fraction derived from the training set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for the train/val split and optimization.",
    )
    parser.add_argument(
        "--train",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "sample_data" / "sample_train.pt"),
        help="Path to the training .pt file (seq_ids, seqs_list, labels_tensor).",
    )
    parser.add_argument(
        "--test",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "sample_data" / "sample_test.pt"),
        help="Path to the test .pt file (seq_ids, seqs_list, labels_tensor).",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=32,
        help="Maximum number of frames per sequence (sequences are padded or truncated to this length).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size used for training and evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of DataLoader workers.",
    )
    parser.add_argument(
        "--pin_memory",
        action="store_true",
        help="Enable pin_memory in DataLoaders.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="mobilenetv4_conv_small",
        choices=[
            "mobilenetv4_conv_small",
            "mobilenetv2_120d",
            "efficientnetv2_b1",
        ],
        help="timm backbone to use for per-frame feature extraction.",
    )

    args = parser.parse_args()

    # Seed everything
    set_seed(args.seed)

    # Set global backbone and Optuna study name
    global TIMM_MODEL_NAME, study_name
    TIMM_MODEL_NAME = args.backbone
    study_name = "BirdClassification-" + TIMM_MODEL_NAME + "-" + SEQ_MODEL_NAME
    print(f"Backbone: {TIMM_MODEL_NAME}")
    print(f"Optuna study name: {study_name}")

    # Load datasets
    full_train_dataset = BirdSequenceDataset(args.train, args.max_seq_len)
    test_dataset = BirdSequenceDataset(args.test, args.max_seq_len)

    # Deterministic stratified train/validation split from the train set
    train_idx, val_idx = get_or_create_split(
        labels=full_train_dataset.labels.numpy(),
        train_pt_path=args.train,
        val_ratio=args.val_ratio,
        seed=args.seed,
        split_dir="splits",
    )

    train_dataset = Subset(full_train_dataset, train_idx)
    val_dataset = Subset(full_train_dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # Determine hyperparameters
    if args.params_json is not None:
        best_params = json.loads(args.params_json)
        print("Using hyperparameters from --params_json.")
    elif args.retrain_best:
        storage_name = f"sqlite:///{study_name}.db"
        db_path = storage_name.replace("sqlite:///", "")
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"Study database not found at {db_path}. "
                "Either run Optuna optimization first or provide --params_json."
            )
        print("Retraining from existing Optuna study best hyperparameters.")
        study = optuna.load_study(study_name=study_name, storage=storage_name)
        best_params = study.best_params
    else:
        best_params = run_optimization(train_loader, val_loader)

    # Train final model and evaluate on the test set
    _ = train_final_model(best_params, train_loader, val_loader)
    evaluate_final_model(test_loader)


if __name__ == "__main__":
    main()
