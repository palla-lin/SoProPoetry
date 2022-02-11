import torch
from torch.utils.data import Dataset

from utils.strings import ENG_CORPUS_PATH, SOP, EOP, EOL, PAD, UNK
from utils.util import read_file


class EnglishPoetryDataset(Dataset):

    def __init__(self, data_path, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.corpus = read_file(data_path)
        self.num_examples = len(self.corpus)

        self._init_vocab()

    def __getitem__(self, index):
        key = f"example_{index}"
        data = self.corpus[key]

        context = data["keywords"]
        
        limit = min(len(context), self.input_size)
        encoded_context = torch.tensor([self.token_to_id[token] for token in context[:limit]])

        output = data["example"]
        limit = min(len(output), self.output_size)
        encoded_output = torch.tensor([self.token_to_id[token] for token in output[:limit]])

        return encoded_context, encoded_output, encoded_context.shape[0], encoded_output.shape[0]

    def __len__(self):
        return self.num_examples

    def _init_vocab(self):
        full_corpus = read_file(ENG_CORPUS_PATH)
        token_id = 5

        self.vocab = {SOP, EOP, EOL, PAD, UNK}
        self.token_to_id = {SOP: 0, EOP: 1, EOL: 2, PAD: 3, UNK: 4}
        self.id_to_token = {0: SOP, 1: EOP, 2: EOL, 3: PAD, 4: UNK}

        for instance in full_corpus.values():
            tokens = instance["keywords"] + instance["example"]

            for token in tokens:
                if token not in self.vocab:
                    self.vocab.add(token)
                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token
                    token_id += 1
        
        self.vocab_size = len(self.vocab)

    def get_token_id(self, token):
        return self.token_to_id[token]

    def get_ids_token(self, id):
        return self.id_to_token[id]
