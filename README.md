<p align="center"><img src="logo_wings_in_time.png" alt="Wings in Time logo" width="220"></p>

# WingNet

Code and example data for *Wings in Time: Robust Classification of Birds from Aerial Motion Sequences* (Nico Klar, Amit Skanda, Leon Marius Schröder, Aamir Ahmad), accepted by IEEE Robotics and Automation Letters (RA-L).

WingNet classifies tracked aerial-object sequences into four classes: **KITE** (0), **BIRD** (1), **AIRCRAFT** (2), and **OTHER** (3). Its convolutional recurrent encoder carries spatial features across frames, with a fourth, normalized time-index channel. This repository includes the model, an evaluation script, a checkpoint, small example train/test files, and auxiliary scripts for selected comparison methods.

## Quick start

Install Python 3.10+ and [Git LFS](https://git-lfs.com/) before cloning so the sample tensors and weights are downloaded, not only their pointer files.

```bash
git clone https://github.com/os-simopt/WingNet.git
cd WingNet
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-wingnet.txt
python wingnet/infer.py --device auto --mode chronological
```

The last command evaluates the supplied checkpoint on 16 example test sequences and prints sequence accuracy, a row-normalized confusion matrix, and per-class/macro F1. It ran successfully here with Python 3.11.8, PyTorch 2.9.1 (CUDA 12.9), and scikit-learn 1.6.1. The sample is **not** the full paper test set; its metrics must not be compared with the paper's reported 93.9%/94.5% results.

For stochastic evaluation (random permutations and their reversals), run:

```bash
python wingnet/infer.py --mode stochastic --permutations 64
```

This mode is slower and needs more GPU memory. A quick functionality check can use `--permutations 2`; `--device cpu` is also supported, but may be slow. Use `--data PATH` and a checkpoint path as the positional argument to evaluate other data/weights; see `python wingnet/infer.py --help`.

## Data and checkpoint

`sample_data/sample_train.pt` contains 40 sequences (10 per class); `sample_data/sample_test.pt` contains 16 (4 per class). Each file is a tuple `(sequence_ids, sequences, labels)`. `sequences[i]` is a `torch.uint8` tensor with shape `(T, 128, 128, 3)` in RGB order; `labels[i]` is an integer class index. `wingnet/weights/model.pt` is the supplied trained checkpoint. The full dataset is intended to be distributed via the [BirdRecorder downloads page](https://birdrecorder.zsw-bw.de/downloads/); as of 2 October 2026, that page does not yet list a WingNet dataset download.

The evaluation script processes stored, complete sequences. Its time index uses each sequence's final length; the released script is therefore an **offline sequence evaluation**, not a demonstrated deployment of live camera/tracker/shutdown hardware.

The `.pt` files are loaded using `torch.load(..., weights_only=True)`. Use only trusted checkpoints and data, and check that any external dataset follows the shape and label convention above.

## Training and auxiliary methods

`wingnet/train.py` contains the two-stage WingNet training code. It requires a CUDA GPU and is configured for long experiments, not a quick demo. Checkpoint selection now uses a deterministic class-stratified validation subset of the training file; the test file is reserved for `infer.py`. Paths are relative to the repository, and new checkpoints go under `wingnet/weights/`. See [WingNet details](wingnet/README.md).

The other folders contain auxiliary implementations: [frame-wise CNN baselines](frame-wise_baselines/README.md), [xLSTM hybrids](lstms/README.md), [MHSA hybrids](mhsa_hybrid_transformers/README.md), and [PGPE](optimizer_pgpe/README.md). These scripts have additional, framework-specific dependencies and were **not** all end-to-end re-run in the publication environment for this release. In particular, the supplied ConvNeXt checkpoint matches Keras `ConvNeXtSmall`, not the paper's named ConvNeXt-V2 baseline; it and the EfficientNet checkpoint give one-class predictions on the example file. See the [baseline status notes](frame-wise_baselines/README.md). This repository should not be interpreted as a complete executable reproduction package for every table row in the article.
