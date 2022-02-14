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


def plot_perplexity(data):
    epochs = list(range(data["epochs"]))
    plt.figure(1)

    # Losses
    plt.subplot(121)
    for loss in ["train_losses", "valid_losses"]:
        lbl = loss.split('_')[0]
        plt.plot(epochs, data[loss], label=lbl)
    plt.ylabel('Loss')
    plt.xlabel("Epoch")
    plt.title('Losses per epoch')
    plt.grid(True)
    leg = plt.legend(loc='best', ncol=1, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    # PPLs
    plt.subplot(122)
    for ppl in ["train_ppls", "valid_ppls"]:
        lbl = ppl.split('_')[0]
        plt.plot(epochs, data[ppl], label=lbl)
    plt.ylabel('PPL')
    plt.xlabel("Epoch")
    plt.title('PPL per epoch')
    plt.grid(True)
    leg = plt.legend(loc='best', ncol=1, mode="expand", shadow=True, fancybox=True)
    leg.get_frame().set_alpha(0.5)

    plt.show()


