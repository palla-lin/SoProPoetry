import time
import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_loaders.eng_dataset import EnglishPoetryDataset
from models.encoder_decoder import EncoderDecoder
from utils.strings import CONFIG_PATH, PAD
from utils.util import read_file, SimpleLossCompute, plot_perplexity, set_seed


def run_epoch(model, dataset, batch_size, loss_compute, print_schedule=10):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    start = time.time()
    total_tokens = 0
    total_loss = 0
    print_tokens = 0

    for index, batch in enumerate(data_loader, 1):
        contexts = torch.stack(batch[0]).cuda()
        poems = torch.stack(batch[1]).cuda()
        num_tokens = (poems != dataset.get_token_id(PAD)).data.sum().item()

        _, _, pre_output = model(contexts, poems)

        loss = loss_compute(pre_output, poems, batch_size)
        total_loss += loss
        total_tokens += num_tokens
        print_tokens += num_tokens
        
        if model.training and index % print_schedule == 0:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (index, loss / batch_size, print_tokens / elapsed))
            start = time.time()
            print_tokens = 0

    return math.exp(total_loss / float(total_tokens))


def train(dataset, config, print_schedule=10):
    num_epochs = config["num_epochs"]
    lr = config["lr"]

    set_seed(config["seed"])

    model = EncoderDecoder(config)
    if config["use_cuda"]:
        model.cuda()

    criterion = nn.NLLLoss(reduction="sum", ignore_index=dataset.get_token_id(PAD))
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    
    dev_perplexities = []

    for epoch in range(num_epochs):
      
        print("Epoch", epoch)
        model.train()
        run_epoch(model,
                  dataset,
                  config["batch_size"],
                  SimpleLossCompute(model.generator, criterion, optim),
                  print_schedule)
        
        model.eval()
        with torch.no_grad():      
            dev_perplexity = run_epoch(model,
                                       dataset,
                                       config["batch_size"], 
                                       SimpleLossCompute(model.generator, criterion, None))
            print("Validation perplexity: %f" % dev_perplexity)
            dev_perplexities.append(dev_perplexity)
        
    return dev_perplexities


if __name__ == "__main__":
    model_config = read_file(CONFIG_PATH)

    dataset = EnglishPoetryDataset(model_config["context_size"], model_config["poem_size"])

    dev_perplexities = train(dataset, model_config)

    plot_perplexity(dev_perplexities)
