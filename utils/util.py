import json
import random

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed):
    """ Set a seed to allow reproducibility. """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def read_file(path):
    extension = path.split('.')[-1].lower()

    if extension == "json":
        with open(path, "r") as json_file:
            data = json.load(json_file)

    return data


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))

    return elapsed_mins, elapsed_secs


def sort_batch(batch, targets, lengths):
    """
    Sort a minibatch by the length of the sequences with the longest sequences first
    return the sorted batch targes and sequence lengths.
    This way the output can be used by pack_padded_sequences(...)
    """
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = batch[perm_idx]
    target_tensor = targets[perm_idx]

    return seq_tensor, target_tensor, seq_lengths


def pad_and_sort_batch(DataLoaderBatch):
    """
    DataLoaderBatch should be a list of (sequence, target, length) tuples...
    Returns a padded tensor of sequences sorted from longest to shortest, 
    """

    batch_size = len(DataLoaderBatch)
    batch_split = list(zip(*DataLoaderBatch))

    cnxts, targs, cnxts_lens, targs_len = batch_split[0], batch_split[1], batch_split[2], batch_split[3]

    cnxts_max = max(cnxts_lens)
    targs_max = max(targs_len)
    padded_cnxts = np.ones((batch_size, cnxts_max)) * 3
    padded_targs = np.ones((batch_size, targs_max)) * 3
    for i, lens in enumerate(zip(cnxts_lens, targs_len)):
        padded_cnxts[i, 0:lens[0]] = cnxts[i][0:lens[0]]
        padded_targs[i, 0:lens[1]] = targs[i][0:lens[1]]

    return sort_batch(torch.tensor(padded_cnxts, dtype=torch.long), torch.tensor(padded_targs, dtype=torch.long), torch.tensor(cnxts_lens, dtype=torch.long))


def plot_perplexity(perplexities):
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)
    plt.show()


class SimpleLossCompute:

    def __init__(self, generator, criterion, optimizer=None):
        self.generator = generator
        self.criterion = criterion
        self.optimizer = optimizer

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
                              y.contiguous().view(-1))
        loss = loss / norm

        if self.optimizer is not None:
            loss.backward()          
            self.optimizer.step()
            self.optimizer.zero_grad()

        return loss.data.item() * norm
