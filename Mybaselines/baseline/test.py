# -*- coding: utf-8 -*-
# Peilu
# January 2022

from model import Seq2seq
import torch.nn as nn
import torch
import pickle
from loader import get_data_loaders
# from config import args
import torch.nn.functional as F
from loguru import logger

logger.add('./output/runtime.log')
logger.info('\n\n\n-------New Record--------\n')

path = './output/best.model'
model_CKPT = torch.load(path)
args = model_CKPT['args']
lang, test_loader = get_data_loaders(('train',), batch_size=1, data_size=args.data_size, build_vocab=False)
model = Seq2seq(vocab_size=lang.n_words, hidden_size=args.hidden_size)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
model.load_state_dict(model_CKPT['state_dict'])
optimizer.load_state_dict(model_CKPT['optimizer'])
device = torch.device('cuda')
model = model.to(device)


def greedy_decode(model, loader, lang):
    model.eval()
    with torch.no_grad():
        for i, (src, tgt, placeholder) in enumerate(loader):
            src = src.to(device)
            tgt = tgt.to(device)
            placeholder = placeholder.to(device)
            enc_output, hidden = model.encoder(src)
            sentence = []
            dec_input = tgt[:, 0].unsqueeze(1)  # <bos>
            for i in range(0, placeholder.size(1)):
                # dec_input = tgt[:, i]
                out, hidden = model.decoder(dec_input, placeholder[:, i].unsqueeze(1), hidden, enc_output)
                out = F.softmax(out, dim=2)
                pred = torch.argmax(out, dim=2)
                dec_input = pred
                sentence.extend(lang.index_to_seq(pred.tolist()[0]))

            for seq in src.tolist():
                seq = lang.index_to_seq(seq)
                # print('gold: ',seq[1:-1])
                logger.info(f'keywords:{seq}')
            for seq in tgt.tolist():
                seq = lang.index_to_seq(seq)
                # print('gold: ',seq[1:-1])
                logger.info(f'gold:{seq[1:-1]}')

            logger.info(f'pred:{sentence[:-1]}\n')


greedy_decode(model, test_loader, lang)


def topk_decode():
    # refer to SongNet
    pass


def report():
    pass
