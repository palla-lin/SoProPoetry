#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun 05 Dec 2021 08:59:31 PM CET

from dataclasses import dataclass
from arguments import parse_arguments

args = parse_arguments()

@dataclass
class Parameters:
    # Preprocessing parameeters
    max_seq_len: int = args.max_seq_len
    num_words: int = 61348

    # Model parameters
    embedding_size: int = args.embedding_size
    bidirectional: bool = True
    save_model: bool = args.save_model
    model_dir: str = args.model_dir
    if args.high_level_tags:
        output_dim: int = 7
    else:  
        output_dim: int = 144
    stride: int = 3
    out_size: int = 32
    
    # Path to pre-trained embeddings
    path: str = args.emb_f
        
    # Training parameters
    hidden_dim: int = 256
    n_layers: int = 3
    learning_rate: float = args.learning_rate
    dropout: float = args.dropout
    epochs: int = args.epochs
    batch_size: int = args.batch_size
    learning_rate: float = args.learning_rate