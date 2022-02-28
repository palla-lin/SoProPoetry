from torch.nn.utils.rnn import pad_sequence
from utils import *
from dataset import *


def collate_fn(batch, test=False):
    for instance in batch:
        inp = instance["input"]
        targ = instance["target"]
        topic = instance["topic"]

        if test:
            tensors = (topic.long(), inp)
            tensors = pad_sequence(tensors, batch_first=True, padding_value=0)
        else:
            tensors = [topic.long(), inp, targ]
            tensors = pad_sequence(tensors, batch_first=True, padding_value=0)
    return tensors


# get split corpora
training, testing, valid = read_file(PATH)
tr_corpus = get_dataset(training)
test_corpus = get_dataset(testing)
val_corpus = get_dataset(valid)

# Dataset
train_dataset = PoemDataset(tr_corpus)
test_dataset = PoemDataset(test_corpus)
val_dataset = PoemDataset(val_corpus)
