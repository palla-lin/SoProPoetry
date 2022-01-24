# -*- coding: utf-8 -*-
# Peilu
# December 2021
import os
import torch.nn as nn
import torch
import torch.nn.functional as F
import random

import wandb
from loguru import logger
from tqdm import tqdm

from loader import get_data_loaders
from config import args
from model3 import Seq2seq
from utils import count_parameters

seed = 1234
random.seed(seed)
torch.manual_seed(seed)

log = './log/'
if not os.path.isdir(log):
    os.mkdir(log)
output = './output/'
if not os.path.isdir(output):
    os.mkdir(output)

# logger.remove()
logger.add('./log/runtime.log', )
logger.info('\n\n\n-------New Record--------\n')

if len(args.gpus.split(',')) >= 2:
    args.parallel = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda')
else:
    args.parallel = False
    device = torch.device('cuda')
if not args.wandb:
    os.environ['WANDB_MODE'] = 'offline'
wandb.init(project="poem")

wandb.config.update(args)  # save arguments and model file
wandb.save('model3.py')

def train(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0
    for i, (src, dec_input, dec_truth, placeholder) in enumerate(train_loader):
        src = src.to(device)
        dec_input = dec_input.to(device)
        dec_truth = dec_truth.to(device)
        placeholder = placeholder.to(device)

        optimizer.zero_grad()
        output = model(src, dec_input, dec_truth, placeholder)
        output = output.reshape(-1, output.size(-1))

        loss = criterion(output, dec_truth.reshape(-1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / (i + 1)


def evaluate(model, criterion, loader):
    eval_loss = 0.
    model.eval()
    with torch.no_grad():
        for i, (src, dec_input, dec_truth, placeholder) in enumerate(loader):
            src = src.to(device)
            dec_input = dec_input.to(device)
            dec_truth = dec_truth.to(device)
            placeholder = placeholder.to(device)

            output = model(src, dec_input, dec_truth, placeholder)
            output = output.reshape(-1, output.size(-1))

            loss = criterion(output, dec_truth.reshape(-1))
            eval_loss += loss.item()
    return eval_loss / (i + 1)
def decode(model, loader, lang):
    model.eval()
    with torch.no_grad():
        for i, (src, dec_input, dec_truth, placeholder) in enumerate(loader):
            src = src.to(device)
            dec_input = dec_input.to(device)
            dec_truth = dec_truth.to(device)
            placeholder = placeholder.to(device)

            output = model(src, dec_input,dec_truth, placeholder)
            output = F.softmax(output, dim=2)
            pred = torch.argmax(output, dim=2)
            for seq in dec_truth:
                indices = seq.tolist()
                seq = lang.index_to_seq(indices)
                # print(seq)
                logger.debug(seq)
                break

            for seq in pred:
                indices = seq.tolist()
                seq = lang.index_to_seq(indices)
                # print(seq)
                logger.debug(seq)
                break
            break
#
# def init_weights(m):
#     for name, param in m.named_parameters():
#         if 'weight' in name:
#             nn.init.normal_(param.data, mean=0, std=0.01)
#         else:
#             nn.init.constant_(param.data, 0)
def main():
    lang, train_loader, valid_loader, test_loader = get_data_loaders(('train', 'valid', 'test'),
                                                                     batch_size=args.batch_size,
                                                                     data_size=args.data_size,
                                                                     min_freq=0)

    model = Seq2seq(vocab_size=lang.n_words,embed_size=args.embed_size,
                    hidden_size=args.hidden_size, num_layers=args.num_layers,dropout=args.dropout)

    # model.apply(init_weights)
    if args.parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    num_para = count_parameters(model)
    logger.info(f'The model has {num_para} trainable parameters')
    # print(f'The model has {num_para} trainable parameters')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    # mile = [20, 150,300]
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=mile, gamma= 0.1)
    best_loss = float('inf')
    observed_worse_val_loss = 0
    observed_worse_val_loss_max = 5
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train(model, optimizer, criterion, train_loader)
        val_loss = evaluate(model, criterion, valid_loader)
        # test_loss = evaluate(model, criterion, test_loader)
        # scheduler.step(val_loss)
        # scheduler.step()
        wandb.log({'train_loss': train_loss, 'val_loss': val_loss})
        logger.info(f'epoch:{epoch}, train_loss:{train_loss}, val_loss:{val_loss}')
        decode(model,valid_loader,lang)
        if val_loss < best_loss:
            best_loss = val_loss
            observed_worse_val_loss = 0
            path = './output/' + "best" + ".model"
            torch.save({'args': args, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
            with open('./output/checkpoint.txt', 'w') as f:
                f.write(f'epoch: {epoch}\nval_loss: {val_loss}\ntrain_loss: {train_loss}\npath: {path}\n')
            logger.info(f'Saved model...')
        elif val_loss > best_loss:
            observed_worse_val_loss += 1
        else:
            pass
        if args.early_stop and observed_worse_val_loss >= observed_worse_val_loss_max:
            logger.info(
                f'Have observed successively {observed_worse_val_loss_max} worse validation results.\nStop training...')
            exit()

#
if __name__ == '__main__':
    main()
