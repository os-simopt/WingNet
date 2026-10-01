import os
from collections import Counter

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import torch

from model.architecture import create_model
import config


def configure_gpus():
    """Enable memory growth on all available GPUs."""
    gpus = tf.config.experimental.list_physical_devices("GPU")
    print("Num GPUs Available:", len(gpus))
    if not gpus:
        return
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print(f"{len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs")
    except RuntimeError as e:
        print("Failed to set memory growth:", e)


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


def load_pt_dataset(path, num_classes=None, selected_indices=None):
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

    if selected_indices is None:
        selected_indices = range(len(inputs))
    for seq_idx in selected_indices:
        sequence = inputs[seq_idx]
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


def stratified_sequence_indices(labels, val_fraction=0.15, seed=42):
    """Split by tracked sequence, preventing adjacent frames crossing sets."""
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_indices, val_indices = [], []
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        if len(indices) < 2:
            raise ValueError(f"Class {label} needs at least two training sequences")
        rng.shuffle(indices)
        n_val = max(1, min(len(indices) - 1, round(len(indices) * val_fraction)))
        val_indices.extend(indices[:n_val].tolist())
        train_indices.extend(indices[n_val:].tolist())
    return train_indices, val_indices


def compute_class_weights(targets_one_hot):
    """Compute class weights from one-hot targets."""
    y_int = np.argmax(targets_one_hot, axis=1)
    class_counts = Counter(y_int)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_int),
        y=y_int,
    )
    class_weight_dict = dict(enumerate(class_weights))

    print("Samples per class:")
    for cls in sorted(class_counts):
        print(f"  class {cls}: {class_counts[cls]} samples")

    print("Class weights:")
    for cls in sorted(class_weight_dict):
        print(f"  class {cls}: {class_weight_dict[cls]:.4f}")

    return class_weight_dict


def build_datagen():
    """Create an ImageDataGenerator with the original augmentation settings."""
    return ImageDataGenerator(
        width_shift_range=20,
        height_shift_range=20,
        shear_range=8.0,
        rotation_range=180,
        brightness_range=[0.5, 1.5],
        zoom_range=0.5,
        rescale=1.0,
        fill_mode="nearest",
    )


def main():
    configure_gpus()

    # Load paths and hyperparameters from config
    train_path = getattr(config, "TRAIN_PATH", "sample_train.pt")
    batch_size = getattr(config, "BATCH_SIZE", 256)
    num_classes = getattr(config, "NUM_CLASSES", 4)
    input_shape = getattr(config, "INPUT_SHAPE", (128, 128, 1))
    weights_path = getattr(
        config,
        "WEIGHTS_PATH",
        os.path.join("weights", "cnn_baseline.weights.h5"),
    )
    lr_schedule = getattr(config, "LR_SCHEDULE", [1e-6, 1e-7])
    max_epochs_per_lr = getattr(config, "MAX_EPOCHS_PER_LR", 1000)
    backbone_name = getattr(config, "BACKBONE", "efficientnetv2b1")

    os.makedirs(os.path.dirname(weights_path), exist_ok=True)

    # Load datasets
    print("Loading training data from:", train_path)
    raw_labels = torch.load(train_path, map_location="cpu", weights_only=True)[2]
    train_indices, val_indices = stratified_sequence_indices(raw_labels)
    train_data, train_targets = load_pt_dataset(
        train_path, num_classes=num_classes, selected_indices=train_indices
    )
    val_data, val_targets = load_pt_dataset(
        train_path, num_classes=num_classes, selected_indices=val_indices
    )
    print(f"Sequence split: {len(train_indices)} training, {len(val_indices)} validation")

    print("train_data:", train_data.shape, "min=", train_data.min(), "max=", train_data.max())
    print("train_targets:", train_targets.shape)
    print("val_data:", val_data.shape, "min=", val_data.min(), "max=", val_data.max())
    print("val_targets:", val_targets.shape)

    class_weight_dict = compute_class_weights(train_targets)

    datagen = build_datagen()
    train_iter = datagen.flow(train_data, train_targets, batch_size=batch_size, shuffle=True)
    steps_per_epoch = max(1, train_data.shape[0] // batch_size)

    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
    print("Number of replicas in sync:", strategy.num_replicas_in_sync)

    with strategy.scope():
        model = create_model()

    print(model.summary())

    best_val_loss = np.inf
    hist_list = []

    # Two-stage training with manual learning-rate schedule
    for i, lr in enumerate(lr_schedule):
        print(f"\n=== Training phase {i + 1}/{len(lr_schedule)} with lr={lr} ===")
        if i > 0 and os.path.exists(weights_path):
            print("Loading best weights from previous phase:", weights_path)
            model.load_weights(weights_path)

        with strategy.scope():
            model.compile(
                loss="categorical_crossentropy",
                optimizer=Adam(learning_rate=lr),
                metrics=["mae", "accuracy"],
            )

        min_epoch_for_phase_end = 10000

        for epoch in range(max_epochs_per_lr):
            history = model.fit(
                train_iter,
                steps_per_epoch=steps_per_epoch,
                validation_data=(val_data, val_targets),
                epochs=1,
                verbose=1,
                class_weight=class_weight_dict,
            )

            val_loss = history.history["val_loss"][-1]
            train_loss = history.history["loss"][-1]
            train_acc = history.history["accuracy"][-1]
            val_acc = history.history["val_accuracy"][-1]
            train_mae = history.history["mae"][-1]
            val_mae = history.history["val_mae"][-1]

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                min_epoch_for_phase_end = max(min_epoch_for_phase_end, int(1.5 * epoch))
                model.save_weights(weights_path)
                print(f"New best val_loss={best_val_loss:.6f} at epoch {epoch}, weights saved.")

                hist_list.append(
                    np.array(
                        [
                            train_loss,
                            val_loss,
                            train_acc,
                            val_acc,
                            train_mae,
                            val_mae,
                        ]
                    )
                )

            if epoch > min_epoch_for_phase_end:
                print(f"Stopping phase {i + 1} early at epoch {epoch}.")
                break

    # Save final model (architecture + weights)
    base, ext = os.path.splitext(weights_path)
    model.save_weights(base + ".weights.h5")
    model.save(base + ".tf.h5", save_format="tf")
    print("Training finished. Best weights stored at:", weights_path)


if __name__ == "__main__":
    main()
