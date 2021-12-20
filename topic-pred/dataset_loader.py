#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun 05 Dec 2021 08:59:31 PM CET


import json
import numpy as np
import pickle

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import LabelEncoder

from utils import *
from arguments import parse_arguments

class DataLoader(object):
    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.args = parse_arguments()
    
    def load_data(self):
        """Load raw poems and its tags
        """
        self.poems = self.data["Poem"].to_numpy()
        self.tags = self.data["Tags"].fillna("").to_numpy()
        self.poem_tags_dict = {}
        for id, (tags, poem) in enumerate(zip(self.tags, self.poems)):
            if tags != '':
                status = "lab"
            else:
                status = "unlab"
            self.poem_tags_dict[id] = {
                "tags": tags,
                "status": status,
                "poem": poem
            }
    
    def get_unique_tags(self):
        """Get list of uniqe tags and save them in a file
        """
        with open(self.args.out_dir+'/uniq_tags.txt', 'w') as f:
            self.tag_freq = unique_tags(self.tags)
            for tag, freq in self.tag_freq.items():
                f.write(tag+"\t"+str(freq)+"\n")
    
    def extract_poems_tags(self):
        """Get all poems and tags, process them and save in text files
        """
        self.poems = []
        self.labels = []
        with open(self.args.out_dir+'/poems.txt', 'w') as fp, open(self.args.out_dir+'/tags.txt', 'w') as ft:
            for id, tag_poem in self.poem_tags_dict.items():
                poem = tag_poem['poem']
                tags = tag_poem['tags']
                proc_poem = pre_process(poem)
                proc_tags = pre_process_tags(tags)
                
                self.poem_tags_dict[id]["poem"] = proc_poem
                if tag_poem['status'] =='lab':
                    self.poem_tags_dict[id]["tags"] = proc_tags
                    hlt_tags = [i for i in proc_tags if self.tag_freq[i]>args.high_level_tags]
                    if len(hlt_tags) != 0:
                        # Self.tags will only contain list of high level tags for each poem
                        self.labels.extend([hlt_tags])
                        self.poem_tags_dict[id]['hlt'] = True
                        self.poem_tags_dict[id]['high_level_tags'] = hlt_tags
                    else:
                        self.poem_tags_dict[id]['hlt'] = False
                
                fp.write(proc_poem+'\n')
                ft.write(",".join(proc_tags) +'\n')
                self.poems.extend([proc_poem])
    
    def tag_mapping(self):
        """Create a tag to integer mapping.
        """
        self.tag2int = {tag: intid+1  for intid, tag in enumerate(list(self.tag_freq.keys()))}
        with open(self.args.out_dir+'/tag2int.json', 'w') as fp:
            json.dump(self.tag2int, fp)
            
    def tokenize(self):
        """Tokenize all poems 
        """
        self.t_words = Tokenizer()
        self.t_words.fit_on_texts(self.poems)  
        self.X = self.t_words.texts_to_sequences(self.poems)

    def build_vocabulary(self):
        """Build a vocabulary of unique words from poems
        """
        self.vocabulary = self.t_words.word_index
        
    def padding(self):
        """Padd varying length of poems
        """
        self.X = pad_sequences(self.X, maxlen=self.args.max_seq_len, truncating='post')
        
    def separate_labeled_unlabeled_poems(self):
        """Self.X  has labeled and unlabeled poems. Here we separate them out.
        """
        self.X_lab, self.X_unlab = [],[]
        for id, tag_poem in self.poem_tags_dict.items():
            if tag_poem['status'] == 'lab':
                if tag_poem['hlt'] == True:
                    self.X_lab.append(self.X[id])
            elif tag_poem['status'] == 'unlab':
                self.X_unlab.append(self.X[id])
    
    def one_hot_encoding(self):
        """Binarize labels
        """
        self.mlb = MultiLabelBinarizer()
        self.y = self.mlb.fit_transform(self.labels)

        
    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_lab, self.y, shuffle=True, 
                                                                                test_size=0.2, random_state=123) 
        self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train, self.y_train, 
                                                                                            shuffle=True, test_size=0.25, 
                                                                                            random_state=123)
    def save_train_test_split(self):
        np.save(self.args.out_dir+'/X_train.npy', self.X_train)
        np.save(self.args.out_dir+'/X_test.npy', self.X_test)
        np.save(self.args.out_dir+'/y_train.npy', self.y_train)
        np.save(self.args.out_dir+'/y_test.npy', self.y_test)
        np.save(self.args.out_dir+'/X_validation.npy', self.X_validation)
        np.save(self.args.out_dir+'/y_validation.npy', self.y_validation)
        with open(self.args.out_dir+'/t_words.pkl', 'wb') as handle:
            pickle.dump(self.t_words, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_train_test_split(self):
        self.X_train =          np.load(self.args.out_dir+'/X_train.npy', allow_pickle=True)
        self.X_test =           np.load(self.args.out_dir+'/X_test.npy', allow_pickle=True)
        self.y_train =          np.load(self.args.out_dir+'/y_train.npy', allow_pickle=True)
        self.y_test =           np.load(self.args.out_dir+'/y_test.npy', allow_pickle=True)
        self.X_validation =     np.load(self.args.out_dir+'/X_validation.npy', allow_pickle=True)
        self.y_validation =     np.load(self.args.out_dir+'/y_validation.npy', allow_pickle=True)
        with open(self.args.out_dir+'/t_words.pkl', 'rb') as fp:
            self.t_words    =       pickle.load(fp)

"""
# Getting a sense of how the tags data looks like
print(yt[0])
print(mlb.inverse_transform(yt[0].reshape(1,-1)))
print(mlb.classes_)
------------------------------------------
Output:
[0 0 0 0 0 0 1 0 0 1]
[('r', 'time series')]
['classification' 'distributions' 'hypothesis testing' 'logistic'
 'machine learning' 'probability' 'r' 'regression' 'self study'
 'time series']
 
"""