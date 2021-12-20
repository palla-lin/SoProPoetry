#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun 05 Dec 2021 08:59:31 PM CET


import numpy as np
import glob, os
import pdb

import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader

from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import average_precision_score as aps

from utils import DatasetMaper
from arguments import parse_arguments
from utils import DEVICE
from parameters import Parameters


class Classify(object):
    def __init__(self, t_words, data, model, params):
        self.t_words = t_words
        self.data = data
        self.model = model
        self.params = params
        self.args = parse_arguments()

    def load_model(self):
        self.model.to(DEVICE)
        model_path = glob.glob(self.params.model_dir + "/*.pth")[0]
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.predict()
    
    
    def predict(self):
        self.model.eval()
        test =  DatasetMaper(self.data['x_test'], self.data['y_test'])
        loader_test = DataLoader(dataset=test, 
                                batch_size=self.params.batch_size, 
                                shuffle=False)
        
        # Start evaluation phase   
        predcited = []
        predcited_prob = []
        with torch.no_grad():
            for x_batch, y_batch in loader_test:
                x_batch = x_batch.long()
                y_batch = y_batch.to(torch.float32)
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                y_pred = self.model(x_batch).cpu().detach()
                y_pred_prob = torch.sigmoid(y_pred).cpu().detach().numpy()
                y_pred = torch.round(torch.sigmoid(y_pred)).cpu().detach().numpy()
                predcited.extend(y_pred)
                predcited_prob.extend(y_pred_prob)

        self.predcited = np.array(predcited)
        self.predcited_prob = np.array(predcited_prob)
        

    
    def evaluate_metric(self):
        themes = []
        y_true = self.data['y_test']
        y_pred = self.predcited
        
        with open(self.args.out_dir+'/uniq_tags.txt', 'r') as f:
            for line in f:
                tag = line.strip().split()[0]
                themes.append(tag)
        themes = themes[:len(y_true[0])]
        print(classification_report(y_true, 
                                    y_pred, 
                                    digits=4,
                                    target_names=themes,
                                    zero_division=0))
        # pdb.set_trace()
        maps = metrics.average_precision_score(y_true, y_pred, average='micro')
        print("Micro Average Precision Score: {:0.4f}".format(maps))
        
        # totalPrecision=0
        # totalSupport=0
        # for i in range (len(themes)):
        #     p = metrics.precision_score(y_true[:,i], y_pred[:,i],zero_division=0)
        #     support= (y_true[:,i]==1).sum()
        #     totalSupport+=support
        #     totalPrecision+= p*support
        #     print("For {} precision: {:.2f} support: {}".format(themes[i], p, support ))
        # print("Weighted Precision: {:.2f}".format(totalPrecision/totalSupport))
        
        
        # f1 = torchmetrics.F1(threshold=0.5, num_classes=None, average='micro')
        # aps = torchmetrics.AveragePrecision(num_classes=129)
        # print(aps(torch.from_numpy(self.predcited_prob), torch.from_numpy(self.data['y_test'])))
        
        
    # def predict(self):
    #     self.model.eval()
    #     test =  DatasetMaper(self.data['x_test'], self.data['y_test'])
    #     loader_test = DataLoader(dataset=test, 
    #                             batch_size=self.params.batch_size, 
    #                             shuffle=False)
        
    #     # multi_labs = np.zeros(shape=(self.data['y_test'].shape), dtype=int)
    #     multi_labs = self.data['y_test']
    #     max_labels = self.data['y_test'].shape[1]
    #     themes = []
    #     with open(self.args.out_dir+'/uniq_tags.txt', 'r') as f:
    #         for line in f:
    #             tag = line.strip().split()[0]
    #             themes.append(tag)
        
    #     all_scores = {} 
    #     # Start evaluation phase
    #     for lab in range(max_labels):
    #         lab_ixs = np.where(multi_labs[:, lab] == 1)[0]
    #         ovr_labs = np.zeros(shape=(multi_labs.shape[0],), dtype=int)
    #         ovr_labs[lab_ixs] = 1

    #         try:
    #             oth_cnt, cur_cnt = np.unique(ovr_labs, return_counts=True)[-1]
    #         except ValueError:
    #             pdb.set_trace()
    #         print_str = "Label {:2d} {:15s} {:4d} {:4d}".format(lab,
    #                                                         themes[lab].upper(),
    #                                                         oth_cnt, cur_cnt)

    #         skf = StratifiedKFold(n_splits=2)
    #         i=0
    #         data = self.data['x_test']
    #         scores = {}
    #         scores = np.zeros(shape=(3, 2), dtype=np.float32)
    #         # pdb.set_trace()
    #         for train_index, test_index in skf.split(data, ovr_labs):

    #             # train_data = [data[i] for i in train_index]
    #             test_data = [data[i] for i in test_index]
                
    #             x_batch = torch.from_numpy(np.array(test_data)).long()
    #             x_batch = x_batch.to(DEVICE)
    #             y_pred = self.model(x_batch).cpu().detach()
                
    #             test_prob = torch.sigmoid(y_pred)
    #             test_pred = torch.round(test_prob).numpy()
                
    #             scores[i][0] = aps(ovr_labs[test_index], test_prob[:, 1])
    #             scores[i][1] = self.compute_f1(ovr_labs[test_index], test_pred)
                
    #             print("\r" + print_str,
    #                     "spl: {:2d}, APS: {:.4f}".format(i, scores[i][0]),
    #                     "F1S: {:.4f}".format(scores[i][1]), end="")

    #             i += 1

    #     all_scores[lab] = scores
    #     print()



            

        

        
    

