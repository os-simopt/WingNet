# wingnet/config.py
"""Central configuration for WingNet.

Paths are relative to this repository, not the caller's working directory.
"""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent

CONFIG = {
    "dataset": {
        "train_path": str(REPOSITORY_ROOT / "sample_data" / "sample_train.pt"),
        "test_path": str(REPOSITORY_ROOT / "sample_data" / "sample_test.pt"),
        "labels_idx": ["Kite", "Bird", "Aircraft", "Other"],
    },

    "model": {
        # Anzahl Feature-Maps im Digest (muss Vielfaches der Eingangs-Kanäle sein)
        "digest_size": 36,
        # Bildgröße der ROIs (128x128)
        "image_size": 128,
        # Anzahl Klassen
        "num_classes": 4,
        # U-Net-Tiefe (Anzahl Down/Up-Stufen)
        "uconv_depth": 4,
        # Dropout-Raten in den Modulen (wie im Original-Code)
        "dropout": {
            "resconv_do1": 0.1,
            "resconv_do2": 0.2,
            "uconv_do1": 0.2,
            "uconv_do2": 0.3,
        },
    },

    "training": {
        # globale Anfangs-Lernrate
        "lr_init": 1e-4,
        "weight_decay": 1e-5,
        "batch_size": 8,
        "min_length": 4,   # minimale Sequenzlänge im Dataset

        # Pretraining-Phase
        "pretrain": {
            "epochs": 4,
            "test_every": 250,
            "c_min": 2,
            "c_max": 6,
        },

        # Haupt-Training
        "main": {
            "epochs": 1000,
            "test_every": 50,
            "c_min": 32,
            "c_max": 128,
        },

        # Exponentielles Sequenz-Gewicht für Loss
        "loss_exp_base": 1.0625,

        # LR-Schedule-Schwellen basierend auf best_avg_trace
        "lr_schedule": {
            "t1": 0.65,
            "t2": 0.8,
            "t3": 0.85,
            "lr1": 5e-5,
            "lr2": 2e-5,
            "lr3": 1e-5,
        },
    },

    "inference": {
        # Anzahl zufälliger Permutationen k für Stochastic Inference (infer.py)
        "stochastic_permutations": 64,
        # Batch-Größe k innerhalb von stochastic_model
        "perm_batch_k": 32,
        # Print alle N Sequenzen im Testlauf
        "print_every": 128,
    },

    "system": {
        # Anzahl CPU-Threads für Torch
        "num_threads": 192,
    },
}

# ===== Abwärtskompatible, bereits im Code verwendete Namen =====

# Datensätze
bird_train_path = CONFIG["dataset"]["train_path"]
bird_test_path = CONFIG["dataset"]["test_path"]

# Klassen-Labels (Mapping wie im Original)
bird_labels = {"Kite": 0, "Bird": 1, "Aircraft": 2, "Other": 3}
bird_label_idx = CONFIG["dataset"]["labels_idx"]

# WingNet Digest-Größe
digest_size = CONFIG["model"]["digest_size"]
