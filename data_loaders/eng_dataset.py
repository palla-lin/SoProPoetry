from dataclasses import dataclass
import json

from torch.utils.data import Dataset

from utils.strings import ENG_CORPUS_PATH, SOP, EOP, EOL, PAD, UNK


class EnglishPoetryDataset(Dataset):

    def __init__(self, input_size=10, output_size=100):
        self.input_size = input_size
        self.output_size = output_size

        with open(f"{ENG_CORPUS_PATH}", 'r') as json_file:
            self.corpus = json.load(json_file)
        self.num_examples = len(self.corpus)

        self._init_vocab()

    def __getitem__(self, index):
        key = f"example_{index}"
        data = self.corpus[key]
        context = data["keywords"]
        output = data["example"]

        return self._prepare(context), self._prepare(output, is_context=False)

    def __len__(self):
        return self.num_examples

    def _init_vocab(self):
        token_id = 5

        self.vocab = {SOP, EOP, EOL, PAD, UNK}
        self.token_to_id = {SOP: 0, EOP: 1, EOL: 2, PAD: 3, UNK: 4}
        self.id_to_token = {0: SOP, EOP: 1, EOL: 2, PAD: 3, UNK: 4}

        for instance in self.corpus.values():
            tokens = instance["keywords"] + instance["example"]

            for token in tokens:
                if token not in self.vocab:
                    self.vocab.add(token)
                    self.token_to_id[token] = token_id
                    self.id_to_token[token_id] = token
                    token_id += 1
        
        self.vocab_size = len(self.vocab)

    def _prepare(self, seq, is_context=True):
        seq_len = len(seq)
        limit = self.input_size if is_context else self.output_size
        
        if seq_len > limit:
            seq = seq[:limit]
        elif seq_len < limit:
            seq += [PAD] * (limit - seq_len)

        return [self.token_to_id[token] for token in seq]

    def get_token_id(self, token):
        return self.token_to_id[token]

    def get_ids_token(self, id):
        return self.id_to_token[id]
