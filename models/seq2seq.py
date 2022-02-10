import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderDecoder(nn.Module):

    def __init__(self, config, pad_idx, device, teacher_forcing=None):
        super().__init__()

        self.embedding = nn.Embedding(config["vocab_size"], 
                                  config["embed_size"], 
                                  padding_idx=pad_idx)
        
        self.dropout = nn.Dropout(config["dropout"])

        self.encoder = Encoder(config["embed_size"], 
                               config["hidden_size"],
                               config["hidden_size"],
                               config["bidirectional"],
                               config["num_layers"],
                               config["dropout"])
        
        self.attention = Attention(config["hidden_size"],
                                   config["hidden_size"],
                                   config["bidirectional"])

        self.decoder = Decoder(config["vocab_size"],
                               config["embed_size"],
                               config["hidden_size"],
                               config["hidden_size"],
                               config["bidirectional"],
                               config["dropout"],
                               self.attention)
        
        self.pad_idx = pad_idx
        self.device = device
        self.teacher_forcing = teacher_forcing if teacher_forcing else config["teacher_forcing"]
        self.vocab_size = config["vocab_size"]
    
    def create_mask(self, cnxt):
        return (cnxt != self.pad_idx).permute(1, 0)

    def forward(self, cnxt, cnxt_lens, target):
        batch_size = target.shape[0]
        target_len = target.shape[1]

        outputs = torch.zeros(target_len, batch_size, self.vocab_size).to(self.device)

        cnxt = cnxt.permute(1, 0)
        embed_cntx = self.dropout(self.embedding(cnxt))

        encoder_outputs, hidden = self.encoder(embed_cntx, cnxt_lens)

        embed_target = self.dropout(self.embedding(target))
        embed_target = embed_target.permute(1, 0, 2)
        decoder_input = embed_target[0]

        mask = self.create_mask(cnxt)

        for t in range(1, target_len):
            output, hidden, _ = self.decoder(decoder_input, hidden, encoder_outputs, mask)
            outputs[t] = output

            to_apply_tf = random.random() < self.teacher_forcing

            top_prediction = output.argmax(1)
            embed_top_prediction = self.dropout(self.embedding(top_prediction))
            decoder_input = embed_target[t] if to_apply_tf else embed_top_prediction

        return outputs


class Encoder(nn.Module):

    def __init__(self, embed_dim, hidden_dim, output_dim, 
                 bidirectional, num_layers, dropout):
        super().__init__()

        self.num_layers = int(num_layers)
        self.bidirectional = bidirectional

        self.rnn = nn.GRU(embed_dim, hidden_dim, self.num_layers,
                          bidirectional=self.bidirectional, dropout=dropout)
        
        linear_input = 2*hidden_dim if bidirectional else hidden_dim
        self.linear = nn.Linear(linear_input, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, embed_cnxt, cnxt_len):
        packed_embed = pack_padded_sequence(embed_cnxt, cnxt_len.to("cpu"))
        packed_outputs, hidden = self.rnn(packed_embed)
        outputs, _ = pad_packed_sequence(packed_outputs)

        if self.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)

        hidden = self.dropout(self.linear(hidden))

        return outputs, hidden


class Attention(nn.Module):

    def __init__(self, encoder_hidden_dim, decoder_hidden_dim, bidirectional):
        super().__init__()

        encoder_hidden_dim = 2*encoder_hidden_dim if bidirectional else encoder_hidden_dim
        self.attn = nn.Linear(encoder_hidden_dim+decoder_hidden_dim, decoder_hidden_dim)
        self.v = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_output, mask):
        cnxt_len = encoder_output.shape[0]

        hidden = hidden.unsqueeze(1).repeat(1, cnxt_len, 1)
        encoder_output = encoder_output.permute(1, 0, 2)

        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_output), dim=2)))
        attention = self.v(energy).squeeze(2)
        attention = attention.masked_fill(mask==0, -1e10)

        return F.softmax(attention, dim=1)


class Decoder(nn.Module):

    def __init__(self, output_dim, embed_dim, encoder_hidden_dim, decoder_hidden_dim, 
                 bidirectional, dropout, attention):
        super().__init__()

        self.output_dim = output_dim
        self.bidirectional = bidirectional
        self.attention = attention

        encoder_hidden_dim = 2*encoder_hidden_dim if bidirectional else encoder_hidden_dim
        self.rnn = nn.GRU(encoder_hidden_dim+embed_dim, decoder_hidden_dim)

        self.linear = nn.Linear(encoder_hidden_dim+decoder_hidden_dim+embed_dim, output_dim)

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, embed_token, hidden, encoder_outputs, mask):
        attn =  self.attention(hidden, encoder_outputs, mask)
        attn = attn.unsqueeze(1)

        encoder_outputs = encoder_outputs.permute(1, 0, 2)

        weighted = torch.bmm(attn, encoder_outputs)
        weighted = weighted.permute(1, 0, 2)
        embed_token = embed_token.unsqueeze(0)

        rnn_input = torch.cat((embed_token, weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        assert (output == hidden).all()

        embed_token = embed_token.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        prediction = self.linear(torch.cat((output, weighted, embed_token), dim=1))

        return prediction, hidden.squeeze(0), attn.squeeze(1)
