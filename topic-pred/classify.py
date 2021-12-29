#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun 05 Dec 2021 08:59:31 PM CET


import numpy as np
import glob, os
import pdb

from nltk.tokenize.treebank import TreebankWordDetokenizer

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader, TensorDataset

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score as aps

from utils import DatasetMaper
from arguments import parse_arguments
from utils import DEVICE
from parameters import Parameters


class Classify(object):
    def __init__(self, t_words, data, model, lb, params):
        self.t_words = t_words
        self.data = data
        self.model = model
        self.params = params
        self.lb = lb
        self.args = parse_arguments()

    def load_model(self):
        self.model.to(DEVICE)
        model_path = glob.glob(self.params.model_dir + "/*.pth")[0]
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.predict()
    
    def predict(self):
        self.model.eval()
        test =  TensorDataset(torch.LongTensor(self.data['x_test']), 
                              torch.LongTensor(self.data['y_test']), 
                              torch.LongTensor(self.data['seq_len_test']))
        loader_test = DataLoader(dataset=test, batch_size=self.params.batch_size, shuffle=False)
        
        # Start evaluation phase
        predcited = []
        corrects = 0
        self.true_vs_pred_dict = {}
        id = 0
        with torch.no_grad():
            for x_batch, y_batch, seq_len in loader_test:
                x_batch = x_batch.long()
                y_batch = y_batch.to(torch.float32)
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                y_pred = self.model(x_batch)
                
                pred_tags = torch.max(y_pred, 1)[1].view(y_batch.size()).data
                corrects += (pred_tags == y_batch.data).sum()
                predcited.extend(pred_tags.cpu().detach().numpy())
                for poem, true_tag, pred_tag in zip(x_batch, y_batch, pred_tags):
                    self.true_vs_pred_dict[id] = {
                        "poem": poem.cpu().numpy(),
                        "true_tag": int(true_tag.item()),
                        "pred_tag": int(pred_tag.item())
                    }
                    id += 1

        self.predcited = np.array(predcited)
        size = len(loader_test.dataset)
        self.accuracy = 100.0 * corrects/size

    def evaluate_metric(self):
        themes = []
        y_true = self.data['y_test']
        y_pred = self.predcited
        print("Test accuracy: {:0.2f}%".format(self.accuracy))
        themes = self.lb.classes_
        print(classification_report(y_true, 
                                    y_pred, 
                                    digits=4,
                                    target_names=themes,
                                    zero_division=0))
    
    def print_some_samples(self):
        i = 0
        for id in self.true_vs_pred_dict:
            enc_poem = self.true_vs_pred_dict[id]["poem"]
            enc_true_tag = self.true_vs_pred_dict[id]["true_tag"]
            enc_pred_tag = self.true_vs_pred_dict[id]["pred_tag"]
            
            dec_poem = self.t_words.sequences_to_texts([enc_poem.tolist()])[0]
            dec_poem_toks = dec_poem.split()
            dec_poem_text = TreebankWordDetokenizer().detokenize(dec_poem_toks)
            dec_true_tag = self.lb.inverse_transform([enc_true_tag]).item()
            dec_pred_tag = self.lb.inverse_transform([enc_pred_tag]).item()
            i+=1
            print("="*20 + "Poem="+str(i)+"="*20)
            print(dec_poem_text)
            print("\nPredicted Tag: ", dec_pred_tag)
            print("Actual Tag: ", dec_true_tag)
            print("="*40,"\n\n")
            
            if i == self.args.print_samples:
                break
