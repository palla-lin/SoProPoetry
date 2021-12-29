#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun 05 Dec 2021 08:59:31 PM CET

"""
Load glove embeddings and create embedding layer
"""

import os
import numpy as np
import torch.nn as nn
import torch
from gensim.models import KeyedVectors
from arguments import parse_arguments
from utils import load_glove_model
args = parse_arguments()


class Embedding(object):
    def __init__(self, params):
        self.path = params.path
        self.args = parse_arguments()
        self.embedding_size = params.embedding_size
    
    def load_embeddings(self, t_words):
        print("\nLoading pre-trained embeddings...")
        if not os.path.exists(self.args.out_dir +'/embedding_weights.npy'):
            print("Embeddings not found. Creating embedding matrix...")
            # self.word2vec = load_glove_model(self.path)
            self.word2vec = KeyedVectors.load_word2vec_format(self.path)
            len_words = len(t_words.word_index) + 1
            self.embedding_weights = np.zeros((len_words, self.embedding_size))
            word2id = t_words.word_index
            for word, index in word2id.items():
                try:
                    self.embedding_weights[index, :] = self.word2vec[word]
                except KeyError:
                    pass
            np.save(self.args.out_dir +'/embedding_weights.npy', self.embedding_weights)
        else:
            self.path_npy = self.args.out_dir +'/embedding_weights.npy'
            self.embedding_weights = np.load(self.path_npy)
        print("--Done--")
        self.create_emb_layer()
        print("--Done--")
        return self.emb_layer
        
    def create_emb_layer(self, non_trainable=False):
        print("\nCreating embedding layer...")
        self.num_embeddings, self.embedding_dim = self.embedding_weights.shape
        self.emb_layer = nn.Embedding(self.num_embeddings, self.embedding_size)
        self.emb_layer.load_state_dict({'weight': torch.Tensor(self.embedding_weights)})
        if non_trainable:
            self.emb_layer.weight.requires_grad = False