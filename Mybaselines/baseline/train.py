# -*- coding: utf-8 -*-
# Peilu
# December 2021
import os
import torch.nn as nn
import torch
import torch.nn.functional as F

import wandb
from loguru import logger
from tqdm import tqdm

from loader import get_data_loaders
from config import args
from model import Seq2seq
from utils import count_parameters

# log= './log/'
# if not os.path.isdir(log):
#     os.mkdir(log)
# output = './output/'
# if not os.path.isdir(output):
#     os.mkdir(output)

logger.add('./output/runtime.log')
logger.info('\n\n\n-------New Record--------\n')
if not args.wandb:
    os.environ['WANDB_MODE'] = 'offline'
wandb.init(project="poetry", config=args.__dict__)

if len(args.gpus.split(',')) >= 2:
    parallel = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda')
else:
    parallel = False
    device = torch.device('cuda')


def train(model, optimizer, criterion, train_loader):
    model.train()
    train_loss = 0
    for i, (src, tgt, placeholder) in enumerate(train_loader):
        src = src.to(device)
        tgt = tgt.to(device)
        placeholder = placeholder.to(device)

        optimizer.zero_grad()
        output = model(src, tgt, placeholder)
        output = output.reshape(-1, output.size(-1))

        gold = tgt[:, 1:].reshape(-1)  # remove first symbol and reshape
        loss = criterion(output, gold)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / (i + 1)


def evaluate(model, criterion, loader):
    eval_loss = 0.
    model.eval()
    with torch.no_grad():
        for i, (src, tgt, placeholder) in enumerate(loader):
            src = src.to(device)
            tgt = tgt.to(device)
            placeholder = placeholder.to(device)
            output = model(src, tgt, placeholder)
            output = output.reshape(-1, output.size(-1))
            gold = tgt[:, 1:].reshape(-1)  # remove first symbol and reshape
            loss = criterion(output, gold)
            eval_loss += loss.item()
    return eval_loss / (i + 1)


def decode(model, loader, lang):
    model.eval()
    with torch.no_grad():
        for i, (src, tgt, placeholder) in enumerate(loader):
            src = src.to(device)
            tgt = tgt.to(device)
            placeholder = placeholder.to(device)
            output = model(src, tgt, placeholder)
            output = F.softmax(output, dim=2)
            pred = torch.argmax(output, dim=2)
            for seq in tgt:
                indices = seq.tolist()
                seq = lang.index_to_seq(indices)
                # print(seq)
                logger.info(seq)
                break

            for seq in pred:
                indices = seq.tolist()
                seq = lang.index_to_seq(indices)
                # print(seq)
                logger.info(seq)
                break
            break


def main():
    lang, train_loader, valid_loader, test_loader = get_data_loaders(('train', 'valid', 'test'),
                                                                     batch_size=args.batch_size,
                                                                     data_size=args.data_size)

    model = Seq2seq(lang.n_words, hidden_size=args.hidden_size, num_layers=args.num_layers)
    if parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    num_para = count_parameters(model)
    logger.info(f'The model has {num_para} trainable parameters')
    # print(f'The model has {num_para} trainable parameters')
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)

    best_loss = float('inf')
    observed_worse_val_loss = 0
    observed_worse_val_loss_max = 5
    for epoch in tqdm(range(1, args.epochs + 1)):
        train_loss = train(model, optimizer, criterion, train_loader)
        val_loss = evaluate(model, criterion, valid_loader)
        # test_loss = evaluate(model, criterion, test_loader)

        wandb.log({'train_loss': train_loss, 'val_loss': val_loss})
        logger.info(f'epoch:{epoch}, train_loss:{train_loss}, val_loss:{val_loss}')
        if epoch % 10 == 0:
            decode(model, train_loader, lang)
        if val_loss < best_loss:
            best_loss = val_loss
            observed_worse_val_loss = 0
            path = './output/' + "best" + ".model"
            torch.save({'args': args, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, path)
            with open('./output/checkpoint.txt', 'w') as f:
                f.write(f'epoch: {epoch}\nval_loss: {val_loss}\npath: {path}\n')
            logger.info(f'Saved model...')
        elif val_loss > best_loss:
            observed_worse_val_loss += 1
        else:
            pass
        if args.early_stop and observed_worse_val_loss >= observed_worse_val_loss_max:
            logger.info(
                f'Have observed successively {observed_worse_val_loss_max} time worse validation results.\nStop training...')
            exit()


if __name__ == '__main__':
    main()
