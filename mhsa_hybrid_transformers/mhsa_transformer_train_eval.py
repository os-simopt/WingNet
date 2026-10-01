# -*- coding: utf-8 -*-
"""
Hybrid CNN + Transformer Model (MHSA) for Aerial Sequence Classification
=======================================================================

This script implements a reproducible training, optimization, and evaluation
pipeline for a hybrid architecture combining a CNN backbone (MobileNetV4) with
a Transformer encoder for temporal modeling of small aerial objects.

The pipeline includes:
1. Loading and preprocessing sequence datasets in .pt format.
2. A hybrid architecture with a frozen CNN backbone and a learnable Transformer.
3. Hyperparameter optimization using Optuna.
4. Final training of the best-performing configuration.
5. Evaluation on an independent test set, including a confusion matrix.

The script is designed for scientific reproducibility:
- Deterministic random seeds.
- Persistent train/validation split regeneration.
- Stored hyperparameter study results (SQLite database).
- Stable model saving/loading conventions.

Dependencies:
    pip install torch torchvision timm optuna scikit-learn matplotlib seaborn tqdm
"""

import os
import json
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


# =============================================================================
# 1. System Setup
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TIMM_MODEL_NAME = "mobilenetv4_conv_small"   # CNN backbone
SEQ_MODEL_NAME = "mhsa"
STUDY_NAME = f"BirdClassification-{TIMM_MODEL_NAME}-{SEQ_MODEL_NAME}"


def set_seed(seed: int = 42):
    """Sets seeds for reproducible model training."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# 2. Deterministic Train/Validation Split
# =============================================================================

def _split_signature(train_pt_path: str, val_ratio: float, seed: int) -> str:
    """Creates a deterministic signature used to store and retrieve split indices."""
    h = hashlib.sha256()
    h.update(os.path.abspath(train_pt_path).encode("utf-8"))
    h.update(str(val_ratio).encode("utf-8"))
    h.update(str(seed).encode("utf-8"))
    return h.hexdigest()[:16]


def get_or_create_split(labels: np.ndarray, *, train_pt_path: str,
                        val_ratio: float, seed: int, split_dir: str = "splits"):
    """Loads an existing split or generates a new stratified train/val split."""
    os.makedirs(split_dir, exist_ok=True)
    signature = _split_signature(train_pt_path, val_ratio, seed)
    split_path = os.path.join(split_dir, f"split_{signature}.json")

    if os.path.exists(split_path):
        with open(split_path, "r") as f:
            data = json.load(f)
        return np.array(data["train_idx"]), np.array(data["val_idx"])

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    all_idx = np.arange(len(labels))
    (train_idx, val_idx), = sss.split(all_idx, labels)

    with open(split_path, "w") as f:
        json.dump({"train_idx": train_idx.tolist(), "val_idx": val_idx.tolist()}, f)

    return train_idx, val_idx


# =============================================================================
# 3. Dataset Definition
# =============================================================================

class BirdSequenceDataset(Dataset):
    """
    Loads sequence-level data stored as .pt tuples:
        (sequence_ids, list_of_sequences, tensor_of_labels)

    Sequences are padded to a fixed length and normalized to [0, 1].
    Each sample returns a tensor of shape (T, C, H, W) and an integer label.
    """

    def __init__(self, pt_file_path: str, max_seq_len: int = 32):
        if not os.path.exists(pt_file_path):
            raise FileNotFoundError(f"Dataset file not found: {pt_file_path}")

        seq_ids, seqs_list, labels_tensor = torch.load(
            pt_file_path, map_location="cpu", weights_only=True
        )

        labels = labels_tensor.numpy()
        valid_idx = np.where(labels != 4)[0]

        self.labels = torch.from_numpy(labels[valid_idx]).long()
        sequences = [seqs_list[i] for i in valid_idx]
        self.padded_sequences = self._pad(sequences, max_seq_len)

    def _pad(self, sequences, max_len):
        padded = []
        for seq in sequences:
            seq_len = seq.shape[0]
            seq = seq.permute(3, 0, 1, 2)  # (T,H,W,C) -> (C,T,H,W)

            if seq_len >= max_len:
                seq = seq[:, :max_len]
            else:
                pad = torch.zeros(seq.shape[0],
                                  max_len - seq_len,
                                  seq.shape[2],
                                  seq.shape[3],
                                  dtype=seq.dtype)
                seq = torch.cat([seq, pad], dim=1)

            padded.append(seq.permute(1, 2, 3, 0))  # back to (T,H,W,C)

        return torch.stack(padded)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.padded_sequences[idx].permute(0, 3, 1, 2).float() / 255.0
        return seq, self.labels[idx]


# =============================================================================
# 4. Hybrid Model (CNN Backbone + Transformer Encoder)
# =============================================================================

class FeatureExtractor(nn.Module):
    """Extracts per-frame CNN features from a frozen timm backbone."""
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        b, t = x.size(0), x.size(1)
        x = x.view(b * t, *x.size()[2:])
        feat = self.backbone(x)[0]
        feat = self.pool(feat)
        return feat.view(b, t, -1)


class HybridModel(nn.Module):
    """Hybrid architecture combining CNN-based frame encoding and MHSA temporal encoding."""
    def __init__(self, hp):
        super().__init__()

        feature_extractor = timm.create_model(
            TIMM_MODEL_NAME, pretrained=True, features_only=True, out_indices=[4]
        )
        for p in feature_extractor.parameters():
            p.requires_grad = False

        self.feature_extractor = FeatureExtractor(feature_extractor)
        feature_dim = 960  # MobileNetV4 conv small

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=hp["transformer_heads"],
            dim_feedforward=hp["mlp_units"] * 2,
            dropout=hp["dropout_rate"],
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer,
                                             num_layers=hp["transformer_layers"])

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(hp["dropout_rate"]),
            nn.Linear(feature_dim, hp["mlp_units"]),
            nn.ReLU(),
            nn.Linear(hp["mlp_units"], 4),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.classifier(x)


# =============================================================================
# 5. Hyperparameter Optimization (Optuna)
# =============================================================================

def objective(trial, train_loader, val_loader):
    hp = {
        "transformer_heads": trial.suggest_categorical("transformer_heads", [4, 8]),
        "transformer_layers": trial.suggest_int("transformer_layers", 1, 3),
        "mlp_units": trial.suggest_categorical("mlp_units", [256, 512]),
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.5),
    }

    model = HybridModel(hp).to(DEVICE)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    opt_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
    if opt_name == "AdamW":
        wd = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()
    best_f1, patience, no_improve = 0.0, 5, 0

    for _ in range(15):
        model.train()
        for seq, lbl in train_loader:
            seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(seq), lbl)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for seq, lbl in val_loader:
                seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
                out = model(seq)
                preds.append(torch.argmax(out, 1).cpu().numpy())
                trues.append(lbl.cpu().numpy())

        f1 = f1_score(np.concatenate(trues), np.concatenate(preds),
                      average="macro", zero_division=0)

        trial.report(f1, _)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if f1 > best_f1:
            best_f1, no_improve = f1, 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    return best_f1


def run_optimization(train_loader, val_loader):
    storage = f"sqlite:///{STUDY_NAME}.db"
    study = optuna.create_study(
        direction="maximize",
        study_name=STUDY_NAME,
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(lambda t: objective(t, train_loader, val_loader), n_trials=100)
    return study.best_params


# =============================================================================
# 6. Final Training
# =============================================================================

def train_final_model(hp, train_loader, val_loader):
    model = HybridModel(hp).to(DEVICE)

    if hp["optimizer"] == "AdamW":
        optimizer = optim.AdamW(model.parameters(),
                                lr=hp["lr"],
                                weight_decay=hp["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=hp["lr"])

    criterion = nn.CrossEntropyLoss()
    best_f1, patience, no_improve = 0.0, 10, 0
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train()
        for seq, lbl in tqdm(train_loader,
                             desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(seq), lbl)
            loss.backward()
            optimizer.step()

        model.eval()
        preds, trues, val_loss_sum = [], [], 0.0
        with torch.no_grad():
            for seq, lbl in tqdm(val_loader,
                                 desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
                out = model(seq)
                val_loss_sum += criterion(out, lbl).item()
                preds.append(torch.argmax(out, 1).cpu().numpy())
                trues.append(lbl.cpu().numpy())

        y_true = np.concatenate(trues)
        y_pred = np.concatenate(preds)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"best_{TIMM_MODEL_NAME}.pth")
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    model.load_state_dict(torch.load(f"best_{TIMM_MODEL_NAME}.pth"))
    return model


# =============================================================================
# 7. Final Evaluation on Test Set
# =============================================================================

def evaluate_final_model(test_loader):
    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_NAME}.db"
    )
    hp = study.best_params
    model = HybridModel(hp).to(DEVICE)
    model.load_state_dict(torch.load(f"best_{TIMM_MODEL_NAME}.pth"))

    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for seq, lbl in test_loader:
            seq, lbl = seq.to(DEVICE), lbl.to(DEVICE)
            out = model(seq)
            preds.append(torch.argmax(out, 1).cpu().numpy())
            trues.append(lbl.cpu().numpy())

    y_true = np.concatenate(trues)
    y_pred = np.concatenate(preds)

    class_names = ["Kite", "Bird", "Aircraft", "Other"]
    print(classification_report(y_true, y_pred,
                                target_names=class_names,
                                zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Confusion Matrix – Hybrid Model (Test Set)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig("confusion_matrix_hybrid_test.png")
    plt.close()


# =============================================================================
# 8. Main Execution
# =============================================================================

def main():
    set_seed(42)

    repository_root = Path(__file__).resolve().parent.parent
    TRAIN_DATA_PATH = str(repository_root / "sample_data" / "sample_train.pt")
    TEST_DATA_PATH = str(repository_root / "sample_data" / "sample_test.pt")

    MAX_SEQ_LEN = 32
    BATCH_SIZE = 16
    VAL_RATIO = 0.15
    SEED = 42

    train_full = BirdSequenceDataset(TRAIN_DATA_PATH, MAX_SEQ_LEN)
    test_dataset = BirdSequenceDataset(TEST_DATA_PATH, MAX_SEQ_LEN)

    train_idx, val_idx = get_or_create_split(
        labels=train_full.labels.numpy(),
        train_pt_path=TRAIN_DATA_PATH,
        val_ratio=VAL_RATIO,
        seed=SEED,
        split_dir="splits",
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(train_full, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=2, pin_memory=True)

    best_params = run_optimization(train_loader, val_loader)
    model = train_final_model(best_params, train_loader, val_loader)
    evaluate_final_model(test_loader)


if __name__ == "__main__":
    main()
