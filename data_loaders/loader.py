# Peilu
# December 2021

import torch
from torch.utils.data import Dataset, DataLoader
import json



class Lang:
    def __init__(self):
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, 'UNK': 3}
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
        src = example['keywords'] + example['placeholder']  # whether to use separator between keywords and placeholder
        src = self.lang.seq_to_index(src)
        tgt = example['content']
        tgt = self.lang.seq_to_index(tgt)

        return src, tgt

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    src, tgt = zip(*data)

    def pad(batch):
        max_length = max([len(seq) for seq in batch])
        for seq in batch:
            while len(seq) < max_length:
                seq.append(0)
        return batch

    src = pad(src)
    src = torch.tensor(src)
    tgt = pad(tgt)
    tgt = torch.tensor(tgt)

    return src, tgt


# temporary design
def preprocess_chinese_data(file_path):
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
                placeholder[-1] = 'C2'
            else:
                content.append(w)
                placeholder.append('C1')
        placeholder[-1] = 'C0'
        placeholder[-1] = 'C3'
        example = {
            'keywords': keywords,
            'content': content,
            'placeholder': placeholder,
        }
        data.append(example)
    # print(data[:3])
    return data


def get_data_loaders(which=('train', 'valid'), batch_size=1):
    train = "../Datasets/CCPC/ccpc_train_v1.0.json"
    valid = "../Datasets/CCPC/ccpc_valid_v1.0.json"
    test = ''
    lang = Lang()
    loaders = []
    for name in which:
        data = preprocess_chinese_data(file_path=vars()[name])
        print(f'Size of {name}:{len(data)}')
        for example in data:
            for seq in example.values():
                lang.add_sequence(seq)

        dataset = Dataset(data=data, lang=lang)
        loader = DataLoader(dataset=dataset, batch_size=batch_size, collate_fn=collate_fn)
        loaders.append(loader)
    print(f'vocab_size:{lang.n_words}')
    if len(loaders) == 1:
        return loaders[0]
    else:
        return loaders


if __name__ == '__main__':
    train_loader, valid_loader = get_data_loaders(('train', 'valid'), batch_size=3)

    for src, tgt in train_loader:
        print(src, src.shape)
        print()
        print(tgt, tgt.shape)
        print()
        break
    for src, tgt in valid_loader:
        print(src, src.shape)
        print()
        print(tgt, tgt.shape)
        print()
        break
