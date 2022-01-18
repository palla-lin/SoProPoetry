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
    
    # Model parameters
    RANDOM_SEED = 73
    MAX_LEN = args.max_len # [default: 1024]

    # hyperparameters
    learning_rate = args.learning_rate
    eps = 1e-8  # epsilon value for the optimizer (prevents potential divide-by-zero errors)
    warmup_steps = 50

    
    # Training parameters
    BATCH_SIZE = args.batch_size    # [default: 2]
    EPOCHS = args.epochs    # [default: 8]
    
    # arguments
    dataset_obj=args.dataset_obj
    out_dir=args.out_dir    # [default: gpt-2]
    save_model=args.save_model  # [default: True]