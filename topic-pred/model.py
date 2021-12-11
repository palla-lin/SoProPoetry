#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun 05 Dec 2021 08:59:31 PM CET

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from embeddings import Embedding

class biLSTM(nn.ModuleList):
    def __init__(self, t_words, params):
        super(biLSTM, self).__init__()
        self.t_words = t_words
        self.embedding_size = params.embedding_size
        self.hidden_dim = params.hidden_dim
        self.n_layers = params.n_layers
        self.dropout = params.dropout
        self.bidirectional = params.bidirectional
        self.output_dim = params.output_dim
        
        # Embedding layer definition
        self.embedding = Embedding(params).load_embeddings(self.t_words)
        
        self.rnn = nn.LSTM(self.embedding_size,
                            self.hidden_dim, 
                            num_layers=self.n_layers, 
                            bidirectional=self.bidirectional, 
                            batch_first=True, 
                            dropout=self.dropout)

        self.fc = nn.Linear(self.hidden_dim*2, self.output_dim)
        self.dropout = nn.Dropout(self.dropout)
        self.act = nn.Sigmoid()
    
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        lstm_out, (hidden, ct) = self.rnn(embedded)
        #concat the final forward and backward hidden state
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        dense_outputs =  self.fc(hidden)
        outputs = self.act(dense_outputs)
        return dense_outputs