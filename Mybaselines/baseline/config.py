# -*- coding: utf-8 -*-
# Peilu
# December 2021

import argparse
import torch

parser = argparse.ArgumentParser(description='proj')
parser.add_argument('-ds', '--data_size',type=float, default=0.01, help='data size')
parser.add_argument('-gpus', '--gpus',type=str, default='0,1', help='set gpu id')
parser.add_argument('-ep', '--epochs', type=int, default=600, help='iteration')
parser.add_argument('-bs', '--batch_size', type=int, default=256, help='iteration')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='learning rate')
parser.add_argument('-dr', '--dropout', type=float, default=0.1, help='dropout rate')
parser.add_argument('-hs', '--hidden_size', type=int, default=300, help='hidden size')
parser.add_argument('-layers', '--num_layers', type=int, default=2, help='num_layers')

parser.add_argument('-es', '--early_stop', type=bool, default=False, help='early_stop')
parser.add_argument('-wandb', '--wandb', type=bool, default=False, help='wandb')

args = parser.parse_args()

