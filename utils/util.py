import json
import random

import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed):
    """ Set a seed to allow reproducibility. """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def read_file(path):
    extension = path.split('.')[-1].lower()

    if extension == "json":
        with open(path, "r") as json_file:
            data = json.load(json_file)

    return data


def plot_perplexity(perplexities):
    plt.title("Perplexity per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Perplexity")
    plt.plot(perplexities)


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
