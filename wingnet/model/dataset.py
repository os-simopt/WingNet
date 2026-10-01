# wingnet/model/dataset.py

import random
import torch
from config import bird_labels


def load_data(path: str, to_load=None):
    """
    loads file, with optional class mask
    """
    with torch.no_grad():
        seq_ids, inputs, targets = torch.load(path, map_location="cpu", weights_only=True)

        seq_list = []
        im_list = []
        t_list = []

        if to_load is None:
            mask = range(4)
        else:
            mask = {bird_labels[s] for s in to_load}

        for idx, seq in enumerate(inputs):
            if targets[idx].item() in mask:
                seq_list.append(seq_ids[idx])
                im_list.append(seq)
                t_list.append(targets[idx].item())

        return seq_list, im_list, torch.tensor(t_list)


class Dataset:
    def __init__(self, inputs: list[torch.Tensor], targets: torch.Tensor, min_length: int):
        self.data = [[], [], [], []]
        self.count = [0, 0, 0, 0]

        # store all items longer than min_length
        with torch.no_grad():
            for idx, obj in enumerate(inputs):
                if obj.shape[0] >= min_length:
                    label = targets[idx].item()
                    self.count[label] += 1
                    self.data[label].append(
                        torch.permute(obj, (0, 3, 1, 2)).to(
                            dtype=torch.float32,
                            memory_format=torch.contiguous_format,
                        )
                        * (1.0 / 255.0)
                    )

        assert min(self.count) != 0

    def make_batches(self, batch_size: int, cut_min: int = 32, cut_max: int = 96):
        """
        makes class balanced set of batches with randomly permuted subsequences
        each batch contains list of sequence, target tensor and list of integer
        tensor denoting chronological indices in original sequence
        """
        ret = []

        new_data = [[], [], [], []]
        count = [0, 0, 0, 0]

        for label, l_data in enumerate(self.data):
            for idx, img_tensor in enumerate(l_data):
                img = img_tensor.clone()

                perm = torch.randperm(img.shape[0])
                img = img[perm]

                while img.shape[0] >= cut_max:
                    s = random.randint(cut_min, cut_max)
                    new_data[label].append((img[-s:].clone(), perm[-s:].clone()))
                    img = img[:-s]
                    perm = perm[:-s]
                    count[label] += 1

                if img.shape[0] >= 2:
                    new_data[label].append((img.clone(), perm.clone()))
                    count[label] += 1

        max_cnt = max(count)

        for label in range(4):
            while count[label] != max_cnt:
                pt = torch.randperm(len(self.data[label]))
                for idx_tensor in pt:
                    idx = idx_tensor.item()

                    img = self.data[label][idx].clone()

                    perm = torch.randperm(img.shape[0])
                    img = img[perm]

                    while img.shape[0] >= cut_max and count[label] != max_cnt:
                        s = random.randint(cut_min, cut_max)
                        new_data[label].append((img[-s:].clone(), perm[-s:].clone()))
                        img = img[:-s]
                        perm = perm[:-s]
                        count[label] += 1

                    if count[label] == max_cnt:
                        break

                    if img.shape[0] >= 2:
                        new_data[label].append((img.clone(), perm.clone()))
                        count[label] += 1

        assert min(count) == max_cnt and max(count) == max_cnt

        n_full_batches = (4 * max_cnt) // batch_size
        shuffle = torch.randperm(4 * max_cnt)
        for i in range(n_full_batches):
            img_list = []
            targ_list = []
            p_l = []
            for j in range(i * batch_size, (i + 1) * batch_size, 1):
                idx = shuffle[j].item()
                label = idx // max_cnt
                idx = idx % max_cnt

                img_list.append(new_data[label][idx][0])
                targ_list.append(torch.zeros(4).scatter_(0, torch.tensor(label), 1.0))
                p_l.append(new_data[label][idx][1])

            ret.append((img_list, torch.stack(targ_list), p_l))

        curr_idx = n_full_batches * batch_size

        if curr_idx != (4 * max_cnt):
            img_list = []
            targ_list = []
            p_l = []
            for j in range(curr_idx, 4 * max_cnt, 1):
                idx = shuffle[j].item()
                label = idx // max_cnt
                idx = idx % max_cnt

                img_list.append(new_data[label][idx][0])
                targ_list.append(torch.zeros(4).scatter_(0, torch.tensor(label), 1.0))
                p_l.append(new_data[label][idx][1])

            ret.append((img_list, torch.stack(targ_list), p_l))

        return ret
