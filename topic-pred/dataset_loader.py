#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @Author: Sangeet Sagar
# @Date:  Sun 05 Dec 2021 08:59:31 PM CET


import json
import numpy as np
import pickle
import random
random.seed(123)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
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
        data_dict = reverse_tag2poem(self.data)
        # Shuffle dict
        l = list(data_dict.items())
        random.shuffle(l)
        data_dict = dict(l)
        
        self.poems = data_dict.keys()
        self.tags = data_dict.values()
        self.poem_tags_dict = {}
        for id, (tags, poem) in enumerate(zip(self.tags, self.poems)):
            stanza_len = len(poem.split("\n\n"))
            if len(tags) > 1:
                pass
            elif stanza_len > 8:
                # Filter out poems longer than 8 stanzas
                pass
            else:
                if tags != '':
                    status = "lab"
                else:
                    status = "unlab"
                self.poem_tags_dict[id] = {
                    "tags": tags[0],
                    "status": status,
                    "poem": poem,
                    "stanza_len": stanza_len
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
        self.stanza_len = []
        with open(self.args.out_dir+'/poems.txt', 'w') as fp, open(self.args.out_dir+'/tags.txt', 'w') as ft:
            for id, tag_poem in self.poem_tags_dict.items():
                poem = tag_poem['poem']
                tags = tag_poem['tags']
                stanza_len = tag_poem['stanza_len']
                proc_poem = pre_process(poem)
                
                self.poem_tags_dict[id]["poem"] = proc_poem
                fp.write(" ".join(proc_poem)+'\n')
                ft.write(tags +'\n')
                self.poems.extend([proc_poem])
                self.labels.extend([tags])
                self.stanza_len.append(stanza_len)
    
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
        
        
    def one_hot_encoding(self):
        """Binarize labels
        """
        # self.mlb = MultiLabelBinarizer()
        self.lb = LabelEncoder()
        self.y = self.lb.fit_transform(self.labels)

        
    def split_data(self):
        # self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, shuffle=True, 
        #                                                                         test_size=0.2, random_state=123) 
        # self.X_train, self.X_validation, self.y_train, self.y_validation = train_test_split(self.X_train, self.y_train, 
                                                                                            # shuffle=True, test_size=0.25, 
                                                                                            # random_state=123)
        self.X_train, self.X_test = custom_train_test_split(self.X)
        self.X_train, self.X_validation = custom_train_test_split(self.X_train)
        
        self.y_train, self.y_test = custom_train_test_split(self.y)
        self.y_train, self.y_validation = custom_train_test_split(self.y_train)
        
        self.seq_len_train, self.seq_len_test = custom_train_test_split(self.stanza_len)
        self.seq_len_train, self.seq_len_validation = custom_train_test_split(self.seq_len_train)

        
    def save_train_test_split(self):
        np.save(self.args.out_dir+'/X_train.npy', self.X_train)
        np.save(self.args.out_dir+'/X_test.npy', self.X_test)
        np.save(self.args.out_dir+'/y_train.npy', self.y_train)
        np.save(self.args.out_dir+'/y_test.npy', self.y_test)
        np.save(self.args.out_dir+'/X_validation.npy', self.X_validation)
        np.save(self.args.out_dir+'/y_validation.npy', self.y_validation)
        
        np.save(self.args.out_dir+'/seq_len_validation.npy', self.seq_len_validation)
        np.save(self.args.out_dir+'/seq_len_train.npy', self.seq_len_train)
        np.save(self.args.out_dir+'/seq_len_test.npy', self.seq_len_test)
        
        with open(self.args.out_dir+'/t_words.pkl', 'wb') as handle:
            pickle.dump(self.t_words, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.args.out_dir+'/lb.pkl', 'wb') as handle:
            pickle.dump(self.lb, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load_train_test_split(self):
        self.X_train =          np.load(self.args.out_dir+'/X_train.npy', allow_pickle=True)
        self.X_test =           np.load(self.args.out_dir+'/X_test.npy', allow_pickle=True)
        self.y_train =          np.load(self.args.out_dir+'/y_train.npy', allow_pickle=True)
        self.y_test =           np.load(self.args.out_dir+'/y_test.npy', allow_pickle=True)
        self.X_validation =     np.load(self.args.out_dir+'/X_validation.npy', allow_pickle=True)
        self.y_validation =     np.load(self.args.out_dir+'/y_validation.npy', allow_pickle=True)
        
        self.seq_len_train =     np.load(self.args.out_dir+'/seq_len_train.npy', allow_pickle=True)
        self.seq_len_test =     np.load(self.args.out_dir+'/seq_len_test.npy', allow_pickle=True)
        self.seq_len_validation =     np.load(self.args.out_dir+'/seq_len_validation.npy', allow_pickle=True)
        
        with open(self.args.out_dir+'/t_words.pkl', 'rb') as fp:
            self.t_words    =       pickle.load(fp)
        with open(self.args.out_dir+'/lb.pkl', 'rb') as fp:
            self.lb    =       pickle.load(fp)
        