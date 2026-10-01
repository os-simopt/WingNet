import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

import config


def get_class_from_id(label_idx: int) -> str:
    return config.CLASS_NAMES.get(int(label_idx), str(label_idx))


def plot_training(hist_list, tag: str, yscale: str = "linear", export: bool = False):
    """
    Plot training and validation loss / accuracy curves.

    hist_list: list of np.array([loss, val_loss, acc, val_acc, mae, val_mae])
    """
    if not hist_list:
        return

    hist_np = np.array(hist_list)
    steps = np.arange(len(hist_np))

    train_loss = hist_np[:, 0]
    val_loss = hist_np[:, 1]
    acc = hist_np[:, 2]
    val_acc = hist_np[:, 3]

    COLOR = "black"
    plt.rcParams["text.color"] = COLOR
    plt.rcParams["axes.labelcolor"] = COLOR
    plt.rcParams["xtick.color"] = COLOR
    plt.rcParams["ytick.color"] = COLOR

    fig, axs = plt.subplots(2, 1, figsize=(18, 18), facecolor="w")

    axs[0].plot(steps, train_loss, label="train_loss")
    axs[0].plot(steps, val_loss, label="val_loss")
    axs[0].grid()
    axs[0].legend(loc="upper right")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].set_yscale(yscale)
    axs[0].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    axs[0].set_title(f"Model Training - {tag}\n", fontsize=16, fontweight="bold")

    axs[1].plot(steps, acc * 100, label="train_acc [%]")
    axs[1].plot(steps, val_acc * 100, label="val_acc [%]")
    axs[1].grid()
    axs[1].legend(loc="upper right")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy [%]")
    axs[1].set_yscale(yscale)
    axs[1].yaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
    axs[1].set_title(f"Model Training - {tag}\n", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if export:
        Path(config.LOG_DIR).mkdir(parents=True, exist_ok=True)
        out_path = Path(config.LOG_DIR) / f"{tag}_training.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")

    plt.show()
    plt.close(fig)
