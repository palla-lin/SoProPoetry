# -*- coding: utf-8 -*-
# Peilu
# December 2021

import torch
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import os
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import copy

class Lang:
    def __init__(self):
        self.word2index = {"<pad>": 0, "<bos>": 1, "<eos>": 2, '<unk>': 3}
        self.n_words = len(self.word2index)  # Count default tokens
        self.index2word = dict([(v, k) for k, v in self.word2index.items()])
        self.counter = Counter()

    def __len__(self):
        return self.n_words

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def count_frequency(self, sequence, identifier='Chinese'):
        if identifier == 'Chinese':
            self.counter.update(sequence)

    def build_vocab(self, counter, min_freq=0):
        for k, v in counter.items():
            if v > min_freq:
                self.add_word(k)

    def seq_to_index(self, sequence):
        indices = [self.word2index[word] if word in self.word2index else self.word2index['<unk>'] for word in sequence]
        return indices

    def index_to_seq(self, indices):
        sequence = [self.index2word[index] for index in indices]
        return sequence


class Dataset(Dataset):
    def __init__(self, data, lang):
        self.data = data
        self.lang = lang

    def __getitem__(self, index):

        example = self.data[index]
        src = example['keywords'] #+ example['placeholder']
        src = self.lang.seq_to_index(src)

        placeholder = copy.deepcopy(example['placeholder'])
        placeholder.append('<eos>')
        placeholder = self.lang.seq_to_index(placeholder)

        dec_input = copy.deepcopy(example['content'])
        dec_input.insert(0, '<bos>')
        dec_input = self.lang.seq_to_index(dec_input)

        dec_truth = copy.deepcopy(example['content'])
        dec_truth.append('<eos>')
        dec_truth = self.lang.seq_to_index(dec_truth)
        return src, dec_input, dec_truth, placeholder

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    src, dec_input, dec_truth, placeholder = zip(*data)

    def pad(batch):
        lengths = [len(seq) for seq in batch]
        max_length = max(lengths)
        for seq in batch:
            while len(seq) < max_length:
                seq.append(0)
        batch = torch.tensor(batch).contiguous()
        return batch

    src = pad(src)
    dec_input = pad(dec_input)
    dec_truth = pad(dec_truth)
    placeholder = pad(placeholder)

    return src, dec_input, dec_truth, placeholder


# temporary design
def preprocess_chinese_data(file_path, data_size=1.0):
    data = []
    for line in open(file_path, 'r', encoding='utf-8'):
        line = json.loads(line)
        keywords = []
        for w in line['keywords']:
            if w != ' ':
                keywords.append(w)
        content = []
        placeholder = []
        for w in line['content']:
            if w == '|':
                placeholder.append('</s>')
                content.append('|')
            else:
                content.append(w)
                placeholder.append('<c1>')

        example = {
            'keywords': keywords,
            'content': content,
            'placeholder': placeholder,
        }
        data.append(example)
    # print(data[:3])
    # exit()
    number = int(len(data) * data_size)
    return data[:number]


def get_data_loaders(which=('train', 'valid'), batch_size=1, data_size=1.0, build_vocab=True, min_freq=0):
    train = "./data/CCPC/ccpc_train_v1.0.json"
    valid = "./data/CCPC/ccpc_valid_v1.0.json"
    test = "./data/CCPC/ccpc_test_v1.0.json"
    lang_path = './output/lang.pkl'
    all_data = {}
    for name in which:
        data = preprocess_chinese_data(file_path=vars()[name], data_size=data_size)
        all_data.update({name: data})
        print(f'{name}:{len(data)}')

    if build_vocab:
        lang = Lang()
        for name in which:
            for examples in all_data[name]:
                for seq in examples.values():
                    lang.count_frequency(seq)
        lang.build_vocab(lang.counter, min_freq=min_freq)
        with open(lang_path, 'wb') as f:
            pickle.dump(lang, f)
    elif os.path.exists(lang_path):
        print(f'read lang from:{lang_path}')
        with open(lang_path, 'rb') as f:
            lang = pickle.load(f)
    else:
        print(f'error, no such path:{lang_path}')
    print(f'vocab_size:{lang.n_words}')
    loaders = [lang]
    for name in which:
        dataset = Dataset(data=all_data[name], lang=lang)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        loaders.append(loader)

    return loaders


if __name__ == '__main__':
    lang, train_loader, valid_loader = get_data_loaders(('train', 'valid'), batch_size=3, build_vocab=True, min_freq=3)
    for src, dec_in,dec_tr , placeholder in train_loader:
        print(dec_in,dec_in.shape, '\n',dec_tr,dec_tr.shape)
        exit()



    pass
