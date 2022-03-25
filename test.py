# poem generation
import math

import torch.nn as nn
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence
import random
from tqdm import tqdm
import torch
from train import PoemGenerator, vocabulary
from train import tr_losses, evaluate_model, NUM_EPOCHS
from data_loader import *
from torch.utils.data import DataLoader
import random


def sample(topic, model):
    with torch.no_grad():

        start_token = ['<BOP>']

        topic_tensor = torch.tensor(topic_voc.lookup_indices([topic]))
        inp_tensor = torch.tensor(vocabulary.lookup_indices(start_token))
        tens = (topic_tensor, inp_tensor)
        topic_tensor, inp_tensor = pad_sequence(tens, batch_first=True, padding_value=0)

        text = []
        text += start_token
        for i in range(1, 25):

            output = model(topic_tensor, inp_tensor)
            #             print('out',output)

            p = functional.softmax(output, dim=-1).data
            #             print("p",p)

            # get indices of top N values
            N = 5  # 3,5,10
            vals, inds = torch.topk(p, N)
            #             print("inds",inds[-1])

            # randomly select one of the three indices
            sample_index = random.sample(inds[-1].tolist(), 1)
            #             print("sampled_index" ,sample_index)

            # get token
            tok = vocabulary.lookup_token(sample_index[0])
            #             print('tok' ,tok)
            if tok == '<EOP>':
                break

            # add to generated text
            text.append(tok + ' ')

            start_token.append(tok)
            inp_tensor = torch.tensor(vocabulary.lookup_indices(start_token))

            tens = (topic_tensor, inp_tensor)
            topic_tensor, inp_tensor = pad_sequence(tens, batch_first=True, padding_value=0)

        print(text[1:])

if __name__ == "__main__":
    vocabulary, topic_voc = train_dataset.build_vocab()
    sample('love',model=PoemGenerator)
    val_dataloader = DataLoader(val_dataset, batch_size=1, collate_fn=collate_fn)
    LOSS_FUNCTION = nn.functional.cross_entropy
    val_losses = []
    for epoch in tqdm(range(NUM_EPOCHS)):
        val_loss=evaluate_model(model=PoemGenerator, dataloader=val_dataloader, loss_function=LOSS_FUNCTION)
        val_losses.append(val_loss)

    # Plots
    plot_loss(tr_losses,val_losses)
    plot_ppl(tr_losses,val_losses)

    # Validation PPL
    ppl_val = math.exp(val_losses[-1])
    print('Validation PPL: ',ppl_val)
