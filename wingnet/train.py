# wingnet/train.py

import torch
import torch.optim
from pathlib import Path

import config
from model.architecture import WingNet
from model.dataset import Dataset, load_data
from model.utils import (
    compute_loss,
    prepare_train_batches,
    prepare_inference,
    compute_metrics,
)


# globale Variablen wie im Original
best_accuracy = 0.0
best_avg_trace = 0.0
lr = config.CONFIG["training"]["lr_init"]
WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"


def split_train_validation(sequences, targets, fraction=0.15, seed=42):
    """Make a repeatable, class-stratified validation split from training data."""
    generator = torch.Generator().manual_seed(seed)
    train_indices = []
    validation_indices = []
    for label in range(4):
        indices = torch.where(targets == label)[0]
        if len(indices) < 2:
            raise ValueError(f"Class {label} needs at least two training sequences")
        indices = indices[torch.randperm(len(indices), generator=generator)]
        n_validation = max(1, min(len(indices) - 1, round(len(indices) * fraction)))
        validation_indices.extend(indices[:n_validation].tolist())
        train_indices.extend(indices[n_validation:].tolist())
    return (
        [sequences[i] for i in train_indices], targets[train_indices],
        [sequences[i] for i in validation_indices], targets[validation_indices],
    )


def main(pretrain: bool):
    global best_accuracy
    global best_avg_trace
    global lr

    best_accuracy = 0.0
    best_avg_trace = 0.0
    lr = config.CONFIG["training"]["lr_init"]
    WEIGHTS_DIR.mkdir(exist_ok=True)

    torch.set_num_threads(config.CONFIG["system"]["num_threads"])

    # Hyperparameter aus config
    weight_decay = config.CONFIG["training"]["weight_decay"]
    batch_size = config.CONFIG["training"]["batch_size"]
    min_length = config.CONFIG["training"]["min_length"]

    if pretrain:
        test_every = config.CONFIG["training"]["pretrain"]["test_every"]
        epochs = config.CONFIG["training"]["pretrain"]["epochs"]

        c_min = config.CONFIG["training"]["pretrain"]["c_min"]
        c_max = config.CONFIG["training"]["pretrain"]["c_max"]
    else:
        test_every = config.CONFIG["training"]["main"]["test_every"]
        epochs = config.CONFIG["training"]["main"]["epochs"]

        c_min = config.CONFIG["training"]["main"]["c_min"]
        c_max = config.CONFIG["training"]["main"]["c_max"]

    # init model
    model_base = WingNet(config.digest_size, 4)

    if not pretrain:
        # Pretraining-Gewichte laden wie bisher
        model_base.load_state_dict(
            torch.load(WEIGHTS_DIR / "pretrained.pt", map_location="cpu", weights_only=True)
        )

    model_base.cuda()
    model = torch.jit.script(model_base)
    model.train()

    # Reserve validation sequences from the training file. The independent test
    # file is used only by infer.py after checkpoint selection.
    with torch.no_grad():
        print("Loading train data...", flush=True)
        _, train_seq, train_targ = load_data(config.bird_train_path)
        train_seq, train_targ, val_seq, val_targ = split_train_validation(
            train_seq, train_targ
        )
        ds = Dataset(train_seq, train_targ, min_length)
        print(f"Train dist.: {ds.count}", flush=True)

        val_ds = prepare_inference(val_seq, val_targ)
        print(f"Validation count: {len(val_ds[0])}", flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # loss exponential weighting
    loss_base = config.CONFIG["training"]["loss_exp_base"]
    weight = loss_base ** torch.linspace(0.0, float(c_max - 1), c_max)

    step = 0
    # main train loop
    for epoch in range(epochs):
        # prepare_train_batches augments with time channel and randomly flips images across width axis
        with torch.no_grad():
            batches = prepare_train_batches(ds.make_batches(batch_size, cut_min=c_min, cut_max=c_max))

        for batch_num, batch in enumerate(batches):
            optimizer.zero_grad()
            loss = compute_loss(model, batch, weight)
            loss.backward()
            optimizer.step()

            # Checkpoint selection uses validation data, never the test file.
            if step != 0 and step % test_every == 0:
                with torch.no_grad():
                    model.eval()
                    acc, mat = compute_metrics(model, val_ds)
                    avg_trace = torch.trace(mat).item() * 0.25

                    if avg_trace > best_avg_trace:
                        torch.save(model_base.state_dict(), WEIGHTS_DIR / "model_trace.pt")
                        best_avg_trace = avg_trace

                    if acc > best_accuracy:
                        torch.save(model_base.state_dict(), WEIGHTS_DIR / "model_accuracy.pt")
                        best_accuracy = acc

                    print()
                    print(f"Validation accuracy: {acc}, best: {best_accuracy}")
                    print(f"Validation avg. trace: {avg_trace}, best: {best_avg_trace}")
                    print("Confusion mat.:")
                    print(mat, flush=True)
                    print()

                    model.train()

                    # update lr based on empirically determined (normalized) accuracy figures
                    sched = config.CONFIG["training"]["lr_schedule"]
                    if sched["t1"] < best_avg_trace <= sched["t2"]:
                        for g in optimizer.param_groups:
                            g["lr"] = sched["lr1"]
                            lr = sched["lr1"]
                    elif sched["t2"] < best_avg_trace <= sched["t3"]:
                        for g in optimizer.param_groups:
                            g["lr"] = sched["lr2"]
                            lr = sched["lr2"]
                    elif sched["t3"] < best_avg_trace:
                        for g in optimizer.param_groups:
                            g["lr"] = sched["lr3"]
                            lr = sched["lr3"]

            # print basic info occasionally
            if step % 10 == 0:
                print(
                    f"Epoch {epoch}, batch {batch_num}, current loss: {loss.item()}",
                    flush=True,
                )

            step += 1

    # save pretrained
    if pretrain:
        torch.save(model_base.state_dict(), WEIGHTS_DIR / "pretrained.pt")


if __name__ == "__main__":
    print("Pretraining...", flush=True)
    main(True)
    print("Pretraining complete", flush=True)

    main(False)
