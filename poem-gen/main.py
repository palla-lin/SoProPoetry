#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun Jan 16 09:08:09 AM CET 2022

"""
<Function of script>
"""


import os
import time
import pdb
import pickle

from dataset_loader import MyDataLoader
from utils import Run, DEVICE, ConditionalGenerate, \
    ConditionalGenerateMultiPoems, \
        ComputePerplexity
from parameters import Parameters as params


def prepare_data(data):
    """ purpose of my function """
    dl = MyDataLoader(data)
    dl.load_data()
    dl.split_data()
    dl.initializer()
    
    return (
        dl.tokenizer, dl.train_dataloader, dl.val_dataloader
    )


def main():
    """ main method """
    os.makedirs(params.out_dir, exist_ok=True)
    # Prepare the data
    print("Loading dataset....")
    data = pickle.load(open(params.dataset_obj, 'rb'))
    tokenizer, train_dataloader, val_dataloader = prepare_data(data)
    # Save tokenizer
    tokenizer.save_pretrained("gpt-2/")
    print("--Done--")
    
    # Train model
    # Run().train(tokenizer, train_dataloader, val_dataloader, params)
    print("\nGenerating poems....")
    # ConditionalGenerate().generate(tokenizer, params)
    # ConditionalGenerateMultiPoems().generate(data, tokenizer, params)
    if params.gen_poem_json:
        ComputePerplexity().compute_ppl(tokenizer, params)
    

if __name__ == "__main__":
    main()