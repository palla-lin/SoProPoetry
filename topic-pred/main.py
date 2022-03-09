#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun 05 Dec 2021 08:59:31 PM CET

"""
<Function of script>
"""

import os
import time
import pdb
import pickle
import pandas as pd

from dataset_loader import DataLoader
from model import CNN
from utils import Run, DEVICE
from parameters import Parameters
from arguments import parse_arguments
from classify import Classify


def prepare_data(data):
    """ purpose of my function """
    dl = DataLoader(data)
    dl.load_data()
    dl.get_unique_tags()
    dl.extract_poems_tags()
    dl.tag_mapping()
    dl.tokenize()
    dl.build_vocabulary()
    dl.padding()
    dl.one_hot_encoding()
    dl.split_data()
    dl.save_train_test_split()
    dl.load_train_test_split()  
    
    return (
        {
            'x_train': dl.X_train,
            'y_train': dl.y_train,
            'x_test': dl.X_test,
            'y_test': dl.y_test,
            'x_valid': dl.X_validation,
            'y_valid': dl.y_validation,
            'seq_len_train': dl.seq_len_train,
            'seq_len_test': dl.seq_len_test,
            'seq_len_valid': dl.seq_len_validation
        },
        dl.t_words, dl.lb
    )

def main():
    """ main method """
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    # Prepare the data
    print("Loading dataset....")
    data = pickle.load(open(args.dataset_obj, 'rb'))
    data, t_words, lb = prepare_data(data)
    print("--Done--")

    # Initialize the model
    start = time.time()
    model = CNN(t_words, Parameters)
    model.to(DEVICE)

    # Train and Evaluate the pipeline
    if not args.trained_model:
        Run().train(model, data, Parameters)
        end = time.time()
        print("*** Training Complete ***")
        print("Training runtime: {:.2f} s".format(end-start))
        # Evaluate on test data
        clf = Classify(t_words, data, model, lb,  Parameters)
        clf.load_model()
        clf.predict()
        clf.evaluate_metric()
        clf.print_some_samples()
    else:
        print("Trained model found")
        print("Using:", args.trained_model)
        # Evaluate on test data
        clf = Classify(t_words, data, model, lb, Parameters)
        clf.load_model()
        clf.predict()
        clf.evaluate_metric()
        clf.print_some_samples()
        
        
if __name__ == "__main__":
    main()