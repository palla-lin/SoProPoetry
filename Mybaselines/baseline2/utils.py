# -*- coding: utf-8 -*-
# Peilu
# December 2021
import torch
import numpy as np
import random
import os


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def seed_torch(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_torch()

#
# def load_checkpoint(model, path, optimizer):
#
#     model_CKPT = torch.load(path)
#     model.load_state_dict(model_CKPT['state_dict'])
#     print('loading checkpoint!')
#     optimizer.load_state_dict(model_CKPT['optimizer'])
#     return model, optimizer
#
# def decode(model, loader, lang):
#     model.eval()
#     with torch.no_grad():
#         for i, (src, tgt, placeholder) in enumerate(loader):
#             src = src.to(device)
#             tgt = tgt.to(device)
#             placeholder = placeholder.to(device)
#             output = model(src, tgt, placeholder)
#             output = F.softmax(output, dim=2)
#             pred = torch.argmax(output, dim=2)
#             for seq in tgt:
#                 indices = seq.tolist()
#                 seq = lang.index_to_seq(indices)
#                 # print(seq)
#                 logger.debug(seq)
#                 break
#
#             for seq in pred:
#                 indices = seq.tolist()
#                 seq = lang.index_to_seq(indices)
#                 # print(seq)
#                 logger.debug(seq)
#                 break
#             break
