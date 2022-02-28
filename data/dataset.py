import torch
from torch.utils.data import Dataset, DataLoader
import torchtext as tt

class PoemDataset(Dataset):
    """
    """
    def __init__(self, corpus):
        self.corpus = corpus
        self.vocab = self.build_vocab()

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
        #         topic_tensor = torch.tensor(self.vocab.lookup_indices([topic]))

        topic_tensor = torch.zeros(1)
        topic_tensor[0] = self.vocab.__getitem__(topic)

        #         print(topic_tensor.type(),topic_tensor.shape)

        sample = {"topic": topic_tensor, "input": inp_tensor, "target": target_tensor}
        return sample

    def build_vocab(self):
        sentences = [poem[1] for poem in self.corpus.values()]

        vocab = tt.vocab.build_vocab_from_iterator(sentences, specials=["<pad>", "<unk>", "<BOP>", "<EOP>"])
        vocab.set_default_index(0)
        return vocab


