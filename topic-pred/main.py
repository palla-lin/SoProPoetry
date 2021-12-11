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
import pandas as pd

from dataset_loader import DataLoader
from model import biLSTM
from utils import Run, DEVICE
from parameters import Parameters
from arguments import parse_arguments
from classify import Classify


def prepare_data(data):
    """ purpose of my function """
    dl = DataLoader(data)
    # dl.load_data()
    # dl.extract_poems()
    # dl.extract_tags()
    # dl.get_unique_tags()
    # dl.tag_mapping()
    # dl.encode_tags()
    # dl.tokenize()
    # dl.build_vocabulary()
    # dl.padding()
    # dl.separate_labeled_unlabeled_poems()
    # dl.one_hot_encoding()
    # dl.split_data()
    # dl.save_train_test_split()
    dl.load_train_test_split()  
    
    return (
        {
            'x_train': dl.X_train,
            'y_train': dl.y_train,
            'x_test': dl.X_test,
            'y_test': dl.y_test,
            'x_valid': dl.X_validation,
            'y_valid': dl.y_validation
        },
        dl.t_words
    )

def main():
    """ main method """
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    # Prepare the data
    print("Loading dataset....")
    data = pd.read_csv(args.csv_data,header=0)
    data, t_words = prepare_data(data)
    print("--Done--")

    # Initialize the model
    start = time.time()
    model = biLSTM(t_words, Parameters)
    model.to(DEVICE)

    # # Train and Evaluate the pipeline
    Run().train(model, data, Parameters)
    end = time.time()
    print("*** Training Complete ***")
    print("Training runtime: {:.2f} s".format(end-start))

    # Evaluate on test data
    clf = Classify(t_words, data, model, Parameters)
    clf.load_model()
    clf.predict()
    clf.evaluate_metric()

if __name__ == "__main__":
    main()