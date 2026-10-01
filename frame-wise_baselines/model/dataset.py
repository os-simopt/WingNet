import numpy as np
import torch
import torch.nn.functional as F

import config


def rgb_to_bayer(img_rgb: np.ndarray, pattern: str = "GRBG") -> np.ndarray:
    """
    Convert an RGB image (H, W, 3) to a synthetic Bayer image (H, W)
    using a GRBG pattern.
    """
    if img_rgb.shape[-1] != 3:
        raise ValueError("Input image must have 3 channels (RGB).")

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


def to_one_hot(label_idx: int, num_classes: int = config.NUM_CLASSES) -> np.ndarray:
    """Convert integer class id to one-hot numpy array."""
    return F.one_hot(torch.tensor(int(label_idx)), num_classes=num_classes).numpy().astype(
        "float32"
    )


def _load_sequence_file(path: str):
    """Load sequence-level .pt file (seq_ids, inputs, targets)."""
    with torch.no_grad():
        seq_ids, inputs, targets = torch.load(path, map_location="cpu", weights_only=True)
    return seq_ids, inputs, targets


def build_frame_dataset(path: str):
    """
    Convert a sequence-level dataset (.pt) into a frame-level dataset.

    For each sequence, all frames are added with the same class label.
    RGB frames are converted to synthetic Bayer images.
    """
    with torch.no_grad():
        seq_ids, inputs, targets = _load_sequence_file(path)

        all_imgs = []
        all_targets = []

        for seq, label in zip(inputs, targets):
            label_oh = to_one_hot(label.item(), num_classes=config.NUM_CLASSES)

            for img in seq:
                if torch.is_tensor(img):
                    img = img.squeeze().cpu().numpy()

                if img.ndim == 3 and img.shape[-1] == 3:
                    img = rgb_to_bayer(img)

                img = img.astype("float32")
                all_imgs.append(img)
                all_targets.append(label_oh)

        data_np = np.stack(all_imgs, axis=0)
        targets_np = np.stack(all_targets, axis=0)

    return data_np, targets_np


def load_train_val():
    """
    Load training and validation data from sample_train.pt and sample_test.pt.

    Returns:
        train_data, train_targets, val_data, val_targets
    """
    train_data, train_targets = build_frame_dataset(str(config.TRAIN_PATH))
    val_data, val_targets = build_frame_dataset(str(config.VAL_PATH))
    return train_data, train_targets, val_data, val_targets
