#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:   2021-02-23 01:52:54
# @Email:  sasa00001@stud.uni-saarland.de
# @Organization: Universität des Saarlandes
# @Last Modified time: 2021-03-18 01:32:15

"""
This script performs topic classification on the automatically 
generated poems and saves the predicted topic for each poem in
a pickled file.
"""

import os
import sys
import argparse
import numpy as np
import re
import json
import pickle
from dataclasses import dataclass
from nltk.tokenize import word_tokenize
import pdb

import torch
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, TensorDataset

from model import CNN
# from utils import pre_process


DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class Parameters:
    # Preprocessing parameeters
    max_seq_len: int = 1000
    num_words: int = 61348

    # Model parameters
    embedding_size: int = 300
    bidirectional: bool = True
    
    output_dim: int = 144
    stride: int = 3
    out_size: int = 32
    dropout: float = 0.15
    out_dir: str="data/NeuralPoet/"
    emb_f: str="../fasttext/wiki-news-300d-1M-subword.vec"


def pre_process(text):
    text = text.replace("[EOL]", "")
    text = text.replace("[EOP]", "")
    text = text.lower()
    text = text.replace("\r", "").strip('\n').strip()
    text = re.sub('\n+', '\n',text)
    text = text.replace("\n ", "")
    # text = text.replace("\n", " <eos> ")
    text = re.sub(' +', ' ',text)
    text = word_tokenize(text)
    return text


def load_gpt2_poems(args):
    """ Load poem generated by gpt-2 """
    all_poems = []
    with open(args.poem_json_fp, 'r') as jf:
        data = json.load(jf)
        
    actual_tags = []
    for tag, line in data.items():
        for id, (kws, ppl_score, poem) in line.items():
            poem = line[id][poem]
            poem = pre_process(poem)
            all_poems.append(poem)
            actual_tags.extend([tag])

    return all_poems, actual_tags

def load_encdec_poems(args):
    """ Load poem generated by encode-decoder model """
    with open(args.poem_json_fp, 'r') as jf:
        data = json.load(jf)
    
    greedy_poem, ktop_poem, actual_tags = [], [], []
    for key, line in data.items():
        gr_poem = pre_process(line["greedy_poem"])
        kt_poem = pre_process(line["ktop_poem"])
        actual_tags.append(line["topic"])
        
        greedy_poem.append(gr_poem)
        ktop_poem.append(kt_poem)
        
    return greedy_poem, ktop_poem, actual_tags


def load_tokenizer_model(args):
    """ purpose of my function """
    # Load tokenizer
    with open(args.tok_fp, 'rb') as fp:
        t_words = pickle.load(fp)
        
    # Load saved model
    model = CNN(t_words, Parameters)
    model.to(DEVICE)
    model.load_state_dict(torch.load(args.model_fp))
    model.eval()
    return t_words, model
    
def tokenize_data(t_words, all_poems):
    X = t_words.texts_to_sequences(all_poems)
    X = pad_sequences(X, maxlen=Parameters.max_seq_len, truncating='post')
    return X

def predict(X, model):
    pred_tags = []
    X = TensorDataset(torch.LongTensor(X))
    X_loader = DataLoader(dataset=X, batch_size=32, shuffle=False)
    with torch.no_grad():
        for poem in X_loader:
            
            poem = poem[0].long()
            poem = poem.to(DEVICE)
            y_pred = model(poem)
            
            pred_tags.extend(
                torch.max(y_pred, 1)[1].data.cpu().detach().numpy()
                )
    
    with open(Parameters.out_dir+'/lb.pkl', 'rb') as fp:
        lb = pickle.load(fp)
    
    
    dec_pred_tag_list = []
    for tag in pred_tags:
        dec_pred_tag = lb.inverse_transform([tag]).item()
        dec_pred_tag_list.extend([dec_pred_tag])
    
    
    return dec_pred_tag_list

def main():
    """ main method """
    args = parse_arguments()
    
    # all_poems, actual_tags = load_gpt2_poems(args)
    # t_words, model = load_tokenizer_model(args)
    # X = tokenize_data(t_words, all_poems)
    # predicted_tags = predict(X, model)
    
    # zip_actual_pred_tags = list(map(list, zip(actual_tags, predicted_tags)))
    # with open(Parameters.out_dir+'/gpt2-topk-poems-pred-tags.pkl', 'wb') as handle:
    #     pickle.dump(zip_actual_pred_tags, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        
    greedy_poem, ktop_poem, actual_tags = load_encdec_poems(args)
    t_words, model = load_tokenizer_model(args)
    X_greedy = tokenize_data(t_words, greedy_poem)
    X_ktop = tokenize_data(t_words, ktop_poem)
    greedy_predicted_tags = predict(X_greedy, model)
    ktop_predicted_tags = predict(X_ktop, model)
    
    to_write = {}
    with open(args.poem_json_fp, 'r') as jf:
        data = json.load(jf)

    for id, (key, line) in enumerate(data.items()):
        # Sanity check
        if actual_tags[id] == data[key]["topic"]:
            data[key].update(
                {
                    "greedy_poem_pred_topic": greedy_predicted_tags[id],
                    "ktop_poem_pred_topic": ktop_predicted_tags[id]
                    
                }
            )
        else:
            print("Error: Mismatch found")
            
    
    with open('../poem-gen/results/encdec_attn-final_poems.json', 'w') as fp:
        json.dump(data, fp, ensure_ascii=False, indent=4)
    
    
    zip_actual_pred_tags = list(map(list, zip(actual_tags, greedy_predicted_tags, ktop_predicted_tags)))
    with open(Parameters.out_dir+'/enc-dec-poems-pred-tags.pkl', 'wb') as handle:
        pickle.dump(zip_actual_pred_tags, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    

def parse_arguments():
    """ parse arguments """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("poem_json_fp", help="path to generated poem json file")
    parser.add_argument("tok_fp", help="file path to with saved tokenizer")
    parser.add_argument("model_fp", help="file path trained model")
    # parser.add_argument("-optional_arg", default=default_value, type=int, help='optional_arg meant for some purpose')
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    main()