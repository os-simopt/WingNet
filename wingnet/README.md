# WingNet implementation

Run commands from the repository root. Install `requirements-wingnet.txt` first and download the LFS files with `git lfs pull`.

## Evaluate the supplied checkpoint

```bash
python wingnet/infer.py
python wingnet/infer.py --mode stochastic --permutations 64
python wingnet/infer.py wingnet/weights/model.pt --data sample_data/sample_test.pt --device cpu
```

The default is chronological evaluation of `sample_data/sample_test.pt` with `wingnet/weights/model.pt`. `--mode both` evaluates both chronological and stochastic variants. A stochastic pass is the sum of predictions over random permutations and their reversals; only the `argmax` matters for the printed metrics. Use `--seed` to repeat a stochastic run. Class order is KITE, BIRD, AIRCRAFT, OTHER.

`infer.py` normalizes the fourth input channel using each **complete** sequence's length. It does not implement a live tracker or turbine-shutdown interface. Reported inference results on the example file are only a smoke test, not the article benchmark.

## Train

```bash
python wingnet/train.py
```

Training requires CUDA and is configured in `wingnet/config.py` for a 4-epoch short-subsequence pretraining phase followed by a 1000-epoch main phase. Each phase uses a class-stratified 85/15 training/validation split derived from `sample_data/sample_train.pt`. During training, validation metrics select checkpoints; the separate test file is not used. Outputs are `wingnet/weights/pretrained.pt`, `model_accuracy.pt`, and `model_trace.pt`. The bundled `model.pt` is left untouched. To evaluate a newly trained checkpoint, pass its path to `infer.py`.

The tiny sample train file is intended for interface checks, not for reproducing the article's full-data results. To train on the complete data, change `CONFIG["dataset"]` paths in `config.py` or use equivalent files in the described tuple format. The full training configuration should be reviewed for the available GPU memory and experiment budget before starting.

## Source map

- `config.py`: model, training, inference and dataset configuration.
- `model/architecture.py`: recurrent encoder, Squeeze-and-Excite blocks and classification head.
- `model/dataset.py`: trusted `.pt` loading, balanced sampling and variable-length subsequences.
- `model/utils.py`: index channel, batches, loss and validation metrics.
- `infer.py`: offline chronological/stochastic evaluation.
- `train.py`: two-stage training and validation-only checkpoint selection.

The two architecture PDFs in this directory are explanatory diagrams, not executable inputs.
