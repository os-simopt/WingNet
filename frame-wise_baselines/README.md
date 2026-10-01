# Frame-wise CNN comparison scripts

This folder contains the supplied Keras/TensorFlow code and checkpoints for single-frame classification of KITE, BIRD, AIRCRAFT, and OTHER. It converts the repository's RGB sequence tensors into synthetic GRBG Bayer-like single-channel frames, then evaluates individual frames. **It does not implement the paper's three sequence-level aggregation rules**, which are reported in the article separately.

The backbone selected in `config.py` is one of `mobilenetv2` (default), `effnetv2b1`, or `convnextsmall`. Despite its legacy filename, `convnextbase.weights.h5` **loads into Keras `ConvNeXtSmall`**, not `ConvNeXtBase` or `ConvNeXtTiny`. This is not the **ConvNeXt-V2** architecture named in the manuscript. Its exact training provenance and preprocessing remain unconfirmed, so this folder **cannot reproduce the paper's ConvNeXt-V2 baseline** without the matching original model definition and data pipeline.

For MobileNetV2, the supplied `mobilenetv2.weights.tf.h5` checkpoint is the working default. The similarly named `mobilenetv2.weights.h5` loads but predicts KITE for every example frame in our check and is kept only for provenance. With the working checkpoint, `infer.py` completed on the example file under TensorFlow 2.17.0/Keras 3.9.0 and yielded 71.6% **frame-level** accuracy over 763 frames; this is not a paper benchmark result.

The supplied `effnetv2b1.weights.h5` loads into the selected Keras model but predicts BIRD for every one of the same 763 example frames (32.2% frame-level accuracy). This may indicate a checkpoint, preprocessing, or domain mismatch; the cause is not established. The checkpoint is retained for provenance, not as a verified reproduction of the published baseline.

With the matching Keras `ConvNeXtSmall` model, `convnextbase.weights.h5` loads but predicts AIRCRAFT for all 763 example frames (28.6% frame-level accuracy). It is likewise retained for provenance only. The legacy filename should not be taken as evidence that it implements the paper's ConvNeXt-V2 comparison.

`config.py` now points to `../sample_data/sample_train.pt` and `../sample_data/sample_test.pt`, independent of the caller's working directory. `infer.py` uses the latter file; `train.py` makes a sequence-level stratified validation split *within the training file*, keeping the test file out of checkpoint selection. Paths to weights and logs are under this folder. Inference supports `--backbone`, `--data`, `--weights`, and `--batch-size` overrides; for training, change `BACKBONE` in `config.py`.

```bash
python frame-wise_baselines/infer.py
python frame-wise_baselines/infer.py --backbone effnetv2b1
python frame-wise_baselines/infer.py --backbone convnextsmall
python frame-wise_baselines/train.py
```

These scripts require PyTorch, NumPy, scikit-learn, and a compatible TensorFlow/Keras installation (including the selected application backbone). The supplied `.weights.h5` files use the Keras 3 weights format; TensorFlow 2.9/Keras 2 could not load them in our check. Only the MobileNetV2 inference path is verified as useful on the example file. The long training workflow has **not** been verified end-to-end in this release environment.
