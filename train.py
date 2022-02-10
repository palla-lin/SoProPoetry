import math
import time

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

from models.seq2seq import EncoderDecoder
from utils.util import read_file, set_seed, init_weights, count_parameters, epoch_time, pad_and_sort_batch
from data_loaders.eng_dataset import EnglishPoetryDataset
from utils.strings import CONFIG_PATH, ENG_TRAIN_PATH, ENG_VALID_PATH, PAD


def train_epoch(model, train_set, batch_size, optimizer, criterion, clip):
    data_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=pad_and_sort_batch)
    model.train()

    epoch_loss = 0
    for batch in tqdm(data_loader):
        contexts = batch[0].cuda()
        poems = batch[1].cuda()
        context_lens = batch[2].cuda()

        optimizer.zero_grad()

        output = model(contexts, context_lens, poems)
        output_dim = output.shape[-1]

        output = output[1:].view(-1, output_dim)
        poems = poems.permute(1, 0)
        poems = poems[1:].contiguous().view(-1)

        loss = criterion(output, poems)
        loss.backward()

        clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)


def eval_epoch(model, valid_set, batch_size, criterion):
    data_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, collate_fn=pad_and_sort_batch)
    model.eval()

    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(data_loader, 1):
            contexts = batch[0].cuda()
            poems = batch[1].cuda()
            context_lens = batch[2].cuda()

            output = model(contexts, context_lens, poems)
            output_dim = output.shape[-1]

            output = output[1:].view(-1, output_dim)
            poems = poems.permute(1, 0)
            poems = poems[1:].contiguous().view(-1)

            loss = criterion(output, poems)
            epoch_loss += loss.item()
    
    return epoch_loss / len(data_loader)


def train(config):
    set_seed(config["seed"])
    num_epochs = config["num_epochs"]
    lr = config["lr"]
    clip = config["clip"]
    batch_size = config["batch_size"]

    train_set = EnglishPoetryDataset(ENG_TRAIN_PATH, config["context_size"], config["poem_size"])
    valid_set = EnglishPoetryDataset(ENG_VALID_PATH, config["context_size"], config["poem_size"])
    pad_idx = train_set.get_token_id(PAD)

    device = torch.device('cuda' if config["use_cuda"] else 'cpu')
    model = EncoderDecoder(config, pad_idx, device).to(device)
    model.apply(init_weights)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = pad_idx)

    best_valid_loss = float('inf')
    for epoch in range(num_epochs):
        start_time = time.time()
    
        train_loss = train_epoch(model, train_set, batch_size, optimizer, criterion, clip)
        valid_loss = eval_epoch(model, valid_set, batch_size, criterion)
        
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'encdec_attn-model.pt')
    
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')


if __name__ == "__main__":
    model_config = read_file(CONFIG_PATH)

    train(model_config)
