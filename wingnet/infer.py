# wingnet/infer.py

import argparse
from pathlib import Path

import torch
from torch import nn
from sklearn.metrics import f1_score

import config
from model.architecture import WingNet
from model.dataset import load_data


def compute_stats(model, inputs: list[torch.Tensor], targets: torch.Tensor):
    """
    Computes and prints stats
    """
    mat = torch.zeros(4, 4)
    s = torch.zeros(4, 4)
    correct = 0

    predictions = torch.empty_like(targets)

    print_every = config.CONFIG["inference"]["print_every"]

    device = next(model.parameters()).device
    for idx, x in enumerate(inputs):
        out = model(x.to(device)).cpu()

        predicted = out[-1].argmax().item()
        t = targets[idx].item()

        mat[t][predicted] += 1.0
        s[t] += 1.0

        predictions[idx] = predicted

        if predicted == t:
            correct += 1

        if idx % print_every == 0:
            print(f"Finished item {idx}", flush=True)

    print("******* Test results *******")
    print(f"Sequence accuracy: {100.0 * correct / len(inputs):.2f}%")
    print(
        "Row-stochastic confusion matrix (rows = ground truth, cols = classifier output):"
    )
    print(mat / s)
    print("Per-class F1-score:")
    print(f1_score(targets, predictions, average=None, zero_division=0))
    print("Macro F1-score:")
    print(f1_score(targets, predictions, average="macro", zero_division=0))
    print(flush=True)


class stochastic_model(nn.Module):
    """
    Performs random permutation inference given WingNet module.
    """

    def __init__(self, module: WingNet, n: int):
        """
        n: number of random permutations to evaluate
        """
        super(stochastic_model, self).__init__()
        self.module = module
        self.n = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # parallel eval batch size - batched eval required due to memory use
        k = config.CONFIG["inference"]["perm_batch_k"]

        r = self.n % k
        its = self.n // k

        ret = torch.zeros((1, 4), device=x.device)

        # each iteration evaluated k random permutations and their respective reversed permutations
        for _ in range(its):
            pl = []
            for _ in range(k):
                p = torch.randperm(x.shape[0])
                pl.append(p)
                pl.append(torch.flip(p, [0]))

            pt = torch.stack(pl).transpose(0, 1)

            d = self.module.init_digest(k + k)

            for p in pt:
                d = self.module.tick(x[p], d)

            ret += torch.sum(self.module.finalize(d), dim=0, keepdim=True)

        # eval rest of batches if n not a multiple of k
        if r != 0:
            pl = []
            for _ in range(r):
                p = torch.randperm(x.shape[0])
                pl.append(p)
                pl.append(torch.flip(p, [0]))

            pt = torch.stack(pl).transpose(0, 1)

            d = self.module.init_digest(r + r)

            for p in pt:
                d = self.module.tick(x[p], d)

            ret += torch.sum(self.module.finalize(d), dim=0, keepdim=True)

        return ret


def main():
    parser = argparse.ArgumentParser(description="Evaluate WingNet on a sequence .pt file.")
    parser.add_argument(
        "model_path", nargs="?", type=Path,
        default=Path(__file__).resolve().parent / "weights" / "model.pt",
        help="WingNet checkpoint (default: wingnet/weights/model.pt)",
    )
    parser.add_argument("--data", type=Path, default=Path(config.bird_test_path))
    parser.add_argument(
        "--mode", choices=("chronological", "stochastic", "both"),
        default="chronological",
        help="Stochastic evaluation is slower and uses random permutations.",
    )
    parser.add_argument("--permutations", type=int,
                        default=config.CONFIG["inference"]["stochastic_permutations"])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.permutations < 1:
        parser.error("--permutations must be at least 1")
    device = torch.device("cuda" if (args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    torch.manual_seed(args.seed)

    with torch.no_grad():
        # load test data and augment time channel
        _, test_seq, test_targ = load_data(str(args.data))
        sequences = []
        targets = []
        img_size = config.CONFIG["model"]["image_size"]

        for idx, seq in enumerate(test_seq):
            if seq.shape[0] < 2:
                raise ValueError(f"Sequence {idx} has fewer than two frames; time index is undefined")
            seq_f = torch.permute(seq, (0, 3, 1, 2)).to(
                dtype=torch.float32, memory_format=torch.contiguous_format
            )
            seq_f *= 1.0 / 255.0

            pt = torch.arange(seq.shape[0]).float() / float(seq.shape[0] - 1)
            seq_f = torch.cat(
                (
                    seq_f,
                    torch.reshape(pt, (seq.shape[0], 1, 1, 1)).repeat(
                        (1, 1, img_size, img_size)
                    ),
                ),
                dim=1,
            )
            sequences.append(seq_f)
            targets.append(test_targ[idx].item())
        test_ds = (sequences, torch.tensor(targets))
        print(f"Test count: {len(test_ds[0])}", flush=True)

        # load model
        m = WingNet(config.digest_size, 4)
        m.load_state_dict(torch.load(args.model_path, map_location="cpu", weights_only=True))
        m.to(device)
        m.eval()

        if args.mode in ("chronological", "both"):
            print("Chronological evaluation")
            compute_stats(m, *test_ds)
        if args.mode in ("stochastic", "both"):
            print(f"Stochastic evaluation ({args.permutations} permutations and reversals)")
            compute_stats(stochastic_model(m, args.permutations), *test_ds)


if __name__ == "__main__":
    main()
