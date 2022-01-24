# -*- coding: utf-8 -*-
# Peilu
# January 2022

from model3 import Seq2seq
import torch.nn as nn
import torch
import pickle
from loader import get_data_loaders
# from config import args
import torch.nn.functional as F
from loguru import logger
logger.info('\n\n\n-------New Record--------\n')
logger.add('./output/result.txt')
device = torch.device('cuda')
path = './output/best.model'
model_CKPT = torch.load(path)
args = model_CKPT['args']
lang, test_loader = get_data_loaders(('train',), batch_size=1, data_size=args.data_size, build_vocab=False)
model = Seq2seq(lang.n_words, embed_size=args.embed_size, hidden_size=args.hidden_size, num_layers=args.num_layers)


if args.parallel:
    model = nn.DataParallel(model).to(device)
else:
    model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.98), eps=1e-9)
model.load_state_dict(model_CKPT['state_dict'])
optimizer.load_state_dict(model_CKPT['optimizer'])

model = model.module



def greedy_decode(model, loader, lang):
    model.eval()
    with torch.no_grad():
        for i, (src, dec_input, dec_truth, placeholder) in enumerate(loader):

            dec_input = dec_input.to(device).transpose(1, 0)
            placeholder = placeholder.to(device).transpose(1, 0)

            enc_output, hidden = model.encoder(src.transpose(1, 0).to(device))
            sentence = []
            dec_inp = dec_input[0, :].unsqueeze(1)  # <bos>

            for i in range(0, placeholder.size(0)):
                placeholder = placeholder[:i + 1, :]
                out, hidden = model.decoder(dec_inp, hidden, placeholder, enc_output)
                out = F.softmax(out, dim=2)
                next_id = torch.argmax(out, dim=2)
                dec_inp = torch.cat((dec_inp,next_id),dim=0)
                next_word = lang.index_to_seq(next_id.tolist()[0])[0]

                if next_word == '<eos>':
                    break
                sentence.extend(next_word)

            for seq in src.tolist():
                seq = lang.index_to_seq(seq)
                seq = ''.join(seq)
                logger.info(f'keywords:{seq}')
            # for seq in dec_truth.tolist():
            #     seq = lang.index_to_seq(seq)
            #     seq = ''.join(seq[:-1])
            #     logger.info(f'gold:{seq}')

            sent = ''.join(sentence)
            logger.info(f'pred:{sent}\n')


def top_k(model, loader, lang):
    model.eval()
    with torch.no_grad():
        for i, (src, dec_input, dec_truth, placeholder) in enumerate(loader):

            dec_input = dec_input.to(device).transpose(1, 0)
            placeholder = placeholder.to(device).transpose(1, 0)

            enc_output, hidden = model.encoder(src.transpose(1, 0).to(device))
            sentence = []
            dec_inp = dec_input[0, :].unsqueeze(1)  # <bos>

            for i in range(0, placeholder.size(0)):
                placeholder = placeholder[:i + 1, :]
                
                out, hidden = model.decoder(dec_inp, hidden, placeholder, enc_output)
                out = F.softmax(out, dim=-1).view(-1)
                pred,idx = torch.topk(out, dim=-1,k=32)
                prob = pred/torch.sum(pred)

                w = torch.multinomial(prob, num_samples=1)

                next_id = idx[w].unsqueeze(0)

                dec_inp = torch.cat((dec_inp, next_id), dim=0)
                next_word = lang.index_to_seq(next_id.tolist()[0])[0]
                if next_word == '<eos>':
                    break
                sentence.extend(next_word)

            for seq in src.tolist():
                seq = lang.index_to_seq(seq)
                seq = ''.join(seq)
                logger.info(f'keywords:{seq}')
            # for seq in dec_truth.tolist():
            #     seq = lang.index_to_seq(seq)
            #     seq = ''.join(seq[:-1])
            #     logger.info(f'gold:{seq}')

            sent = ''.join(sentence)
            logger.info(f'pred:{sent}\n')

if __name__ == '__main__':
    # logger.info('\n----------decoded by greedy decode-----------\n')
    # greedy_decode(model, test_loader, lang)
    logger.info('\n----------decoded by topk-----------\n')
    top_k(model, test_loader, lang)

    # greedy_decode(model, test_loader, lang)