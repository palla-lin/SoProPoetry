#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun Jan 16 09:22:12 AM CET 2022


import numpy as np
import pickle
import random

import torch
from torch.utils.data import Dataset, random_split, \
    DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2Tokenizer


from utils import *
from parameters import Parameters as params
from arguments import parse_arguments

RANDOM_SEED=123
random.seed(RANDOM_SEED)

class NeuralPoetDataset(Dataset):
    """Load poem data into a customized Dataset 
    object that inherits from PyTorch's Dataset class.

    Args:
        Dataset ([type]): [description]
    """

    def __init__(self, data, tokenizer, max_length, gpt2_type='gpt2'):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        for i in data:
            encodings_dict = tokenizer('<BOS>' + i + '<EOS>',
                                       truncation=True,
                                       max_length=max_length,
                                       padding='max_length')

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(
                encodings_dict['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]


class MyDataLoader(object):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.args = parse_arguments()
        self.MAX_LEN = params.MAX_LEN
        self.BATCH_SIZE = params.BATCH_SIZE

    def load_data(self):
        """Load raw poems and its tags
        """
        self.poems, self.tags, self.stanza_len = [], [], []
        for _, line in self.data.items():
            self.poems.append(line["poem"])
            self.tags.append(line["tags"])
            self.stanza_len.append(line["stanza_len"])

        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        special_tokens_dict = {
            'bos_token': '<BOS>',  # beginning of sents
            'eos_token': '<EOS>',  # end of sents
            'pad_token': '<PAD>'}  # padding toks
        num_added_tokens = self.tokenizer.add_special_tokens(special_tokens_dict)

        self.poem_dataset = NeuralPoetDataset(
            self.poems, self.tokenizer, max_length=self.MAX_LEN)

    def split_data(self):
        """Split dataset into train, test and validation set
        """
        def train_val_split(split, dataset):
            train_size = int(split * len(dataset))
            val_size = len(dataset) - train_size
            return train_size, val_size
        
        train_size, val_size =  train_val_split(0.8, self.poem_dataset)
        self.X_train, self.X_valid = random_split(self.poem_dataset, [train_size, val_size])


    def initializer(self):
        """Divide dataset into individual batches for training
        using DataLoeader class
        """
        self.train_dataloader = DataLoader(self.X_train,
                                           sampler=RandomSampler(self.X_train),
                                           batch_size=self.BATCH_SIZE)

        # No need to randomize data since validation set is only
        # needed to evaluate the model
        self.val_dataloader = DataLoader(self.X_valid,
                                         sampler=SequentialSampler(self.X_valid),
                                         batch_size=self.BATCH_SIZE)
    