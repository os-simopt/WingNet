# wingnet/model/utils.py

import random
import torch
import config

IMG_SIZE = config.CONFIG["model"]["image_size"]


def parallel_forward(model, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    evals list of sequences (that may be of different lengths),
    while trying to maximize parallelism of hardware
    """
    n = len(inputs)
    full_size = config.digest_size + 4

    buffer = torch.empty(
        (n, full_size, IMG_SIZE, IMG_SIZE),
        device=inputs[0].device,
        dtype=inputs[0].dtype,
    )
    buffer[:, 4:, :, :] = model.get_init()

    indices: list[tuple[int, int]] = []
    results: list[list[torch.Tensor]] = []
    ret: list[torch.Tensor] = []

    for idx, x in enumerate(inputs):
        if not x.shape[0]:
            raise Exception("empty input")
        buffer[idx, :4] = x[0]
        indices.append((idx, 1))
        results.append([])
        ret.append(torch.Tensor())

    while True:
        digest = model.fast_tick(buffer)

        new_indices = []
        new_input = []

        for idx, (x_p, x_i) in enumerate(indices):
            d = digest[idx]
            results[x_p].append(d)

            if x_i == inputs[x_p].shape[0]:
                ret[x_p] = model.finalize(torch.stack(results[x_p])).cpu()
                results[x_p].clear()
            else:
                new_input.append(inputs[x_p][x_i])
                new_input.append(d)
                new_indices.append((x_p, x_i + 1))

        if not new_indices:
            break

        indices = new_indices
        buffer = torch.cat(new_input).view((len(new_indices), full_size, IMG_SIZE, IMG_SIZE))

    return ret


def compute_loss(
    model,
    batch: tuple[list[torch.Tensor], torch.Tensor],
    weight: torch.Tensor,
) -> torch.Tensor:
    """
    computes loss for training
    """
    inputs, targets = batch

    c_in = []
    for in_t in inputs:
        c_in.append(in_t.cuda())

    y_list = parallel_forward(model, c_in)

    ret = torch.zeros(1)

    for idx, y in enumerate(y_list):
        n = y.shape[0]
        w = weight[:n]
        loss = torch.nn.functional.cross_entropy(
            y,
            targets[idx].unsqueeze(0).repeat(n, 1),
            reduction="none",
        )
        ret += torch.dot(w, loss) / (w.sum())

    return ret / float(len(y_list))


def compute_metrics(model, batch: tuple[list[torch.Tensor], torch.Tensor]):
    """
    computes and returns raw accuracy and normalized confusion matrix
    """
    inputs, targets = batch
    bits = torch.zeros(len(inputs), 4)
    mat = torch.zeros(4, 4)
    s = torch.zeros(4, 4)

    for idx, x in enumerate(inputs):
        out = model.simple_forward(x.cuda()).cpu()
        y = torch.max(out[-1], dim=0)[1]
        t = torch.max(targets[idx], dim=0)[1]

        mat[t][y] += 1.0
        s[t] += 1.0
        bits[idx][y] = 1.0

    return 1.0 - (((bits - targets).abs().sum().item() * 0.5) / float(len(inputs))), mat / s


def prepare_train_batches(batches):
    """
    augments list of batches with fourth time channel and randomly
    flips batch items across width channel
    """
    prepared = []
    for batch in batches:
        inp_l = []
        for idx, x in enumerate(batch[0]):
            # convert chronological indices in original sequence to relative indices in subsequence
            pl = batch[2][idx]
            pt = torch.argsort(torch.argsort(pl, dim=0), dim=0).float() / float(pl.shape[0] - 1)

            if random.randrange(0, 2) == 1:
                mf = torch.cat(
                    (
                        torch.flip(x, [3]),
                        torch.reshape(pt, (x.shape[0], 1, 1, 1)).repeat(
                            (1, 1, IMG_SIZE, IMG_SIZE)
                        ),
                    ),
                    dim=1,
                )
            else:
                mf = torch.cat(
                    (
                        x,
                        torch.reshape(pt, (x.shape[0], 1, 1, 1)).repeat(
                            (1, 1, IMG_SIZE, IMG_SIZE)
                        ),
                    ),
                    dim=1,
                )

            inp_l.append(mf)

        prepared.append((inp_l, batch[1]))

    return prepared


def prepare_inference(sequences, targets):
    """
    prepares test dataset for intermittent testing while training
    - performs time channel augmentation and one hot class encoding
    """
    prepared_seq = []
    prepared_targ = []
    for idx, x in enumerate(sequences):
        pt = torch.arange(x.shape[0]).float() / float(x.shape[0] - 1)
        vx = (
            torch.permute(x, (0, 3, 1, 2))
            .to(dtype=torch.float32, memory_format=torch.contiguous_format)
            * (1.0 / 255.0)
        )
        prepared_seq.append(
            torch.cat(
                (
                    vx,
                    torch.reshape(pt, (x.shape[0], 1, 1, 1)).repeat(
                        (1, 1, IMG_SIZE, IMG_SIZE)
                    ),
                ),
                dim=1,
            )
        )

        vt = torch.zeros(4)
        vt[targets[idx]] = 1.0
        prepared_targ.append(vt)

    return prepared_seq, torch.stack(prepared_targ)
