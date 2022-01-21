from enum import Enum

import torch
import numpy as np


class DecoderStrategy(Enum):
    GREEDY = 1


class TextDecoder:

    def __init__(self, decoder_type=DecoderStrategy.GREEDY):
        self.decoder_type = decoder_type

    def decode(self, model):
        if self.decoder_type is DecoderStrategy.GREEDY:
            return self._greedy_decode(model)

    def _greedy_decode(self, model, context, max_len, sos_index=0, eos_index=1):
        with torch.no_grad():
            encoder_hidden, _ = model.encode(context)
            prev_y = torch.ones(1, 1).fill_(sos_index).type_as(context)

        output = []
        hidden = None

        for _ in range(max_len):
            with torch.no_grad():
                _, hidden, pre_output = model.decode(prev_y, encoder_hidden, hidden)
                prob = model.generator(pre_output[:, -1])

            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.data.item()
            output.append(next_word)
            prev_y = torch.ones(1, 1).type_as(context).fill_(next_word)
        
        output = np.array(output)
            
        # cut off everything starting from </s> 
        # (only when eos_index provided)
        if eos_index is not None:
            first_eos = np.where(output==eos_index)[0]
            if len(first_eos) > 0:
                output = output[:first_eos[0]]      
        
        return output
