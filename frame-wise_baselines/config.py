"""
Configuration for frame-wise CNN baseline.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Model / data parameters
# ---------------------------------------------------------------------------

# Allowed backbone identifiers:
#   "effnetv2b1"
#   "mobilenetv2"
#   "convnextsmall" (the checkpoint filename retains a legacy "base" label)
BACKBONE = "mobilenetv2"

NUM_CLASSES = 4

INPUT_HEIGHT = 128
INPUT_WIDTH  = 128
INPUT_CHANNELS = 1
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, INPUT_CHANNELS)

# Regularization & dropout (used in architecture.py)
WEIGHT_DECAY = 1e-8
DROPOUT = 0.35

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent

TRAIN_PATH = ROOT.parent / "sample_data" / "sample_train.pt"
VAL_PATH   = ROOT.parent / "sample_data" / "sample_test.pt"

WEIGHTS_DIR = ROOT / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)

# The supplied MobileNetV2 checkpoint with the `.tf.h5` suffix matches this
# Keras implementation. The other similarly named file gives degenerate
# all-KITE predictions on the example test data and is kept for provenance.
WEIGHTS_FILE = (
    "mobilenetv2.weights.tf.h5" if BACKBONE == "mobilenetv2"
    else "convnextbase.weights.h5" if BACKBONE == "convnextsmall"
    else f"{BACKBONE}.weights.h5"
)
WEIGHTS_PATH = WEIGHTS_DIR / WEIGHTS_FILE

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

BATCH_SIZE      = 256     # for training
EVAL_BATCH_SIZE = 16      # for inference

# Learning-rate schedule for multi-phase training
LR_SCHEDULE = [1e-6, 1e-7]
MAX_EPOCHS_PER_LR = 1000
