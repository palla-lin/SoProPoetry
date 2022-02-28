# poem generation
from torch.nn import functional
from torch.nn.utils.rnn import pad_sequence
import random
import torch
from train import PoemGenerator, vocabulary

def sample(topic ,model):

    with torch.no_grad():

        vocab = vocabulary
        start_token = "<BOP>"

        topic_tensor = torch.zeros(1)
        topic_tensor[0] = vocab.__getitem__(topic)
        topic_tensor = topic_tensor.long()
        #         print('topic old',topic_tensor)

        inp_tensor = torch.zeros(1)
        inp_tensor[0] = vocab.__getitem__(start_token)
        inp_tensor = inp_tensor.long()
        #         print('inp tensor old',inp_tensor)


        text = []
        for i in range(1 ,50):
            output, _ = model(topic_tensor, inp_tensor)


            p = functional.softmax(output, dim=1).data
            print(p)

            # get indices of top N values

            N = 10  # 3,5
            vals, inds = torch.topk(p ,N)
            print("top_n_ind" ,inds)

            # randomly select one of the three indices
            sample_index = random.sample(inds[0].tolist() ,1)
            print("sampled_token_index" ,sample_index)

            # get token
            tok = vocab.lookup_token(sample_index[0])
            print('tok' ,tok)

            # add to generated text
            text.append(tok +' ')

            inp_tensor = torch.cat((inp_tensor, torch.zeros(1).long()), 0)
            inp_tensor[i] = sample_index[0]
            print("inp tensor new", inp_tensor)

            tens = (topic_tensor, inp_tensor)
            topic_tensor, inp_tensor = pad_sequence(tens, batch_first=True, padding_value=0)

            text.append(tok + ' ')
        print(text)

if __name__ == "__main__":
    sample('alone',model=PoemGenerator)