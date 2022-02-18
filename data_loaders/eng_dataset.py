import torch
from torch.utils.data import Dataset

from utils import strings
from utils.util import read_file


class EnglishPoetryDataset(Dataset):

    def __init__(self, data_path, input_size, output_size, use_glove):
        self.input_size = input_size
        self.output_size = output_size
        self.corpus = read_file(data_path)
        self.num_examples = len(self.corpus)

        self._init_vocab(use_glove)

    def __getitem__(self, index):
        key = f"example_{index}"
        data = self.corpus[key]

        context = data["keywords"]
        
        limit = min(len(context), self.input_size)
        encoded_context = torch.tensor([self.get_token_id(token) for token in context[:limit]])

        output = data["example"]
        limit = min(len(output), self.output_size)
        encoded_output = torch.tensor([self.get_token_id(token) for token in output[:limit]])

        return encoded_context, encoded_output, encoded_context.shape[0], encoded_output.shape[0]

    def __len__(self):
        return self.num_examples

    def _init_vocab(self, use_glove):
        if use_glove:
            self.token_to_id = read_file(strings.TOKEN_TO_ID_GLOVE_PATH)
            self.id_to_token = read_file(strings.ID_TO_TOKEN_GLOVE_PATH)
            self.glove_embeddings = read_file(strings.GLOVE_EMBED_PATH)
        else:
            self.token_to_id = read_file(strings.TOKEN_TO_ID_PATH)
            self.id_to_token = read_file(strings.ID_TO_TOKEN_PATH)
    
        self.vocab_size = len(self.token_to_id)

    def get_token_id(self, token):
        if token not in self.token_to_id:
            return self.token_to_id[strings.UNK]

        return self.token_to_id[token]

    def get_ids_token(self, id):
        return self.id_to_token[id]

    def get_glove(self):
        return self.glove_embeddings

    def get_vocab_size(self):
        return self.vocab_size
