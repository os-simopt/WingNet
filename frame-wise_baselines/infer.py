import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import torch

from model.architecture import create_model
import config


def rgb_to_bayer(img_rgb, pattern="GRBG"):
    """
    Convert an RGB image (H, W, 3) into a synthetic Bayer pattern (H, W).
    Matches the GRBG logic used in the training notebook.
    """
    if img_rgb.ndim != 3 or img_rgb.shape[-1] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3).")

    h, w, _ = img_rgb.shape
    bayer = np.zeros((h, w), dtype=img_rgb.dtype)

    for y in range(h):
        for x in range(w):
            if y % 2 == 0:
                if x % 2 == 0:
                    bayer[y, x] = img_rgb[y, x, 1]  # G
                else:
                    bayer[y, x] = img_rgb[y, x, 0]  # R
            else:
                if x % 2 == 0:
                    bayer[y, x] = img_rgb[y, x, 2]  # B
                else:
                    bayer[y, x] = img_rgb[y, x, 1]  # G

    return bayer


def load_pt_dataset(path, num_classes=None):
    """
    Load WingNet-style .pt file and flatten sequences to frame-level samples.

    Expected format:
        (seq_ids: list, inputs: list of sequences, targets: tensor [N_sequences])
    """

    # Load raw object
    obj = torch.load(path, map_location="cpu", weights_only=True)

    if not (isinstance(obj, tuple) and len(obj) == 3):
        raise ValueError(f"Expected (seq_ids, inputs, targets), got {type(obj)}")

    seq_ids, inputs, targets = obj

    # Convert targets to numpy integer labels
    if hasattr(targets, "numpy"):
        targets = targets.numpy()
    targets = np.asarray(targets).astype("int64")

    # Frame-level containers
    all_frames = []
    all_labels = []

    for seq_idx, sequence in enumerate(inputs):
        label = targets[seq_idx]

        for frame in sequence:
            if torch.is_tensor(frame):
                frame = frame.squeeze().cpu().numpy()

            # Convert RGB → Bayer if needed
            if frame.ndim == 3 and frame.shape[-1] == 3:
                frame = rgb_to_bayer(frame)

            all_frames.append(frame.astype("float32"))
            all_labels.append(label)

    # Stack into array
    X = np.stack(all_frames, axis=0)
    y_int = np.asarray(all_labels)

    # Prepare channels: (H,W) → (H,W,1)
    if X.ndim == 3:
        X = X[..., np.newaxis]

    # Normalize to [0,1] if necessary
    if X.max() > 1.0:
        X = X / 255.0

    # One-hot encoding
    if num_classes is None:
        num_classes = int(y_int.max()) + 1

    y = tf.keras.utils.to_categorical(y_int, num_classes=num_classes)

    return X.astype("float32"), y.astype("float32")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a frame-wise CNN on example frames.")
    parser.add_argument(
        "--backbone", choices=("mobilenetv2", "effnetv2b1", "convnextsmall"),
        default=config.BACKBONE,
    )
    parser.add_argument("--data", type=Path, default=config.VAL_PATH)
    parser.add_argument("--weights", type=Path, help="Override the selected backbone checkpoint")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    config.BACKBONE = args.backbone
    default_weights = (
        "mobilenetv2.weights.tf.h5" if args.backbone == "mobilenetv2"
        else "convnextbase.weights.h5" if args.backbone == "convnextsmall"
        else f"{args.backbone}.weights.h5"
    )
    weights_path = args.weights or config.WEIGHTS_DIR / default_weights

    print("Loading evaluation data from:", args.data)
    val_data, val_targets = load_pt_dataset(args.data, num_classes=config.NUM_CLASSES)
    print("val_data:", val_data.shape, "min=", val_data.min(), "max=", val_data.max())
    print("val_targets:", val_targets.shape)

    model = create_model()
    print("Loading weights from:", weights_path)
    model.load_weights(weights_path)

    preds = model.predict(val_data, batch_size=args.batch_size, verbose=0)
    y_true = np.argmax(val_targets, axis=1)
    y_pred = np.argmax(preds, axis=1)

    acc = (y_true == y_pred).mean()
    print(f"\nAccuracy: {acc:.4f}")

    print("\nConfusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))


if __name__ == "__main__":
    main()
