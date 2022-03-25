import torch
from torch.utils.data import Dataset, DataLoader
import torchtext as tt


class PoemDataset(Dataset):

    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab, self.topic_vocab = self.build_vocab()

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        topic = self.corpus[idx][0]
        poem = self.corpus[idx][1]
        inp = poem[:-1]
        target = poem[1:]

        # convert sentences to tensors
        inp_tensor = torch.tensor(self.vocab.lookup_indices(inp))
        target_tensor = torch.tensor(self.vocab.lookup_indices(target))
        topic_tensor = torch.tensor(self.topic_vocab.lookup_indices(topic))
        sample = {"topic": topic_tensor, "input": inp_tensor, "target": target_tensor}
        return sample

    def build_vocab(self):
        sentences = [poem[1] for poem in self.corpus.values()]

        vocab = tt.vocab.build_vocab_from_iterator(sentences, specials=["<pad>", "<unk>", "<BOP>", "<EOP>"])
        vocab.set_default_index(0)

        topics = [poem[0] for poem in self.corpus.values()]
        topic_vocab = tt.vocab.build_vocab_from_iterator(topics, specials=["<pad>", "<unk>"])
        topic_vocab.set_default_index(0)

        return vocab, topic_vocab





