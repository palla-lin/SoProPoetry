# -*- coding: utf-8 -*-
# Peilu
# December 2021

import torch
from torch.utils.data import Dataset, DataLoader
import json
import pickle
import os


class Lang:
    def __init__(self):
        self.word2index = {"<pad>": 0, "<bos>": 1, "<eos>": 2, '<unk>': 3}
        self.n_words = len(self.word2index)  # Count default tokens
        self.index2word = dict([(v, k) for k, v in self.word2index.items()])

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def add_sequence(self, sequence, identifier='Chinese'):
        if identifier == 'Chinese':
            for token in sequence:
                self.add_word(token)

    def seq_to_index(self, sequence):
        indices = [self.word2index[word] if word in self.word2index else self.word2index['UNK'] for word in sequence]
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
        src = example['keywords']  # + example['placeholder']
        src = self.lang.seq_to_index(src)

        placeholder = self.lang.seq_to_index(example['placeholder'])
        tgt = example['content']
        tgt = self.lang.seq_to_index(tgt)

        return src, tgt, placeholder

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    src, tgt, placeholder = zip(*data)

    def pad(batch):
        max_length = max([len(seq) for seq in batch])
        for seq in batch:
            while len(seq) < max_length:
                seq.append(0)
        return batch

    src = pad(src)
    src = torch.tensor(src).contiguous()
    tgt = pad(tgt)
    tgt = torch.tensor(tgt).contiguous()
    placeholder = pad(placeholder)
    placeholder = torch.tensor(placeholder).contiguous()

    return src, tgt, placeholder


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
        content.insert(0, '<bos>')
        content.append('<eos>')
        placeholder.append('<eos>')
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


def get_data_loaders(which=('train', 'valid'), batch_size=1, data_size=1.0, build_vocab=True):
    train = "./data/CCPC/ccpc_train_v1.0.json"
    valid = "./data/CCPC/ccpc_valid_v1.0.json"
    test = "./data/CCPC/ccpc_test_v1.0.json"
    lang_path = './output/lang.pkl'

    if build_vocab:
        lang = Lang()
    elif os.path.exists(lang_path):
        print(f'read lang from:{lang_path}')
        with open(lang_path, 'rb') as f:
            lang = pickle.load(f)
    else:
        print(f'error, no such path:{lang_path}')
    loaders = [lang]

    for name in which:
        data = preprocess_chinese_data(file_path=vars()[name], data_size=data_size)
        print(f'Size of {name}:{len(data)}')
        if build_vocab:
            for example in data:
                for seq in example.values():
                    lang.add_sequence(seq)

        dataset = Dataset(data=data, lang=lang)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
        loaders.append(loader)
    if build_vocab:
        with open(lang_path, 'wb') as f:
            pickle.dump(lang, f)
    print(f'vocab_size:{lang.n_words}')

    return loaders


if __name__ == '__main__':
    lang, train_loader, valid_loader = get_data_loaders(('train', 'valid'), batch_size=3)
    for src, tgt, placeholder in train_loader:
        print(src, src.shape, tgt.shape, placeholder.shape)
        exit()

    #     print()
    #     print(tgt, tgt.shape)
    #     print()
    #     break
    # for src, tgt in valid_loader:
    #     print(src, src.shape)
    #     print()
    #     print(tgt, tgt.shape)
    #     print()
    #     break
    # for i, (src,tgt) in enumerate(train_loader):
    #     print(i)
    #     print(src.shape)
    #     print(tgt.shape)
    #     break

    pass
