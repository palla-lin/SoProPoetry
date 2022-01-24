# -*- coding: utf-8 -*-
"""
@Author: Peilu
@Location: Germany
@Date: 2022/1/23 
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=200,enc_hidden_size=1024, enc_num_layers=2, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(enc_hidden_size * 2)
        self.gru = nn.GRU(embed_size, enc_hidden_size, num_layers=enc_num_layers, bidirectional=True)

    def forward(self, x):

        embedded = self.dropout(self.embedding(x))
        self.gru.flatten_parameters()
        out, hidden = self.gru(embedded)
        out = self.layer_norm(out)
        # hidden = hidden[-1] + hidden[-2]
        # out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]

        # out(length, batch_size, hidden_size)
        # hidden(layers, batch_size, hidden_size)
        return out, hidden#[-1].unsqueeze(0)


class AttDecoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size=200,enc_hidden_size=1024,dec_hidden_size=1024, num_layers=2, dropout=0.1):
        super(AttDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

        self.attn = Attention(enc_hidden_size, dec_hidden_size)
        self.gru = nn.GRU(embed_size, dec_hidden_size, num_layers=num_layers,bidirectional=True)

        self.layer_norm = nn.LayerNorm(dec_hidden_size + enc_hidden_size*2)
        self.linear2 = nn.Linear(dec_hidden_size+ enc_hidden_size * 2, vocab_size)

    def forward(self, tgt, hidden, placeholder, enc_output):
        tgt_emb = self.embedding(tgt)
        place_emb = self.embedding(placeholder)
        tgt_emb_dropout = self.dropout(tgt_emb+place_emb)
        self.gru.flatten_parameters()
        output, hidden = self.gru(tgt_emb_dropout, hidden)
        s = hidden[-1] + hidden[-2]
        attn = self.attn(s, enc_output).unsqueeze(1)
        c = torch.bmm(attn, enc_output.transpose(1, 0)).transpose(0, 1)

        all = self.layer_norm(torch.cat((c,s.unsqueeze(0)),dim=2))
        output = self.linear2(all)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):

        # s = s[-1,:,:]  # only the last layer
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]

        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]

        # s = [batch_size, src_len, dec_hid_dim]
        # enc_output = [batch_size, src_len, enc_hid_dim ]
        s = s.unsqueeze(1).repeat(1, src_len, 1)
        enc_output = enc_output.transpose(0, 1)

        # energy = [batch_size, src_len, dec_hid_dim]
        energy = torch.tanh(self.attn(torch.cat((s, enc_output), dim=2)))
        # attention = [batch_size, src_len]
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)


class Seq2seq(nn.Module):
    def __init__(self, vocab_size,embed_size, hidden_size, num_layers=2, dropout=0.1):
        super(Seq2seq, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = EncoderRNN(vocab_size,embed_size, hidden_size, num_layers, dropout)
        self.decoder = AttDecoderRNN(vocab_size, embed_size,hidden_size, hidden_size, num_layers, dropout)
        self.teacher_forcing = 1.0

    def forward(self, src, dec_input, dec_truth, placeholder):
        # src(batch_size,keyword_length)
        # dec_input(batch_size,<bos>+body_length)
        # dec_truth(batch_size,body_length+<eos>)
        # placeholder(batch_size,body_length+<eos>)
        src = src.transpose(1, 0)
        tgt = dec_input.transpose(1, 0)
        placeholder = placeholder.transpose(1, 0)

        enc_output, hidden = self.encoder(src)
        outputs = torch.zeros(placeholder.size(0), placeholder.size(1), self.vocab_size).to('cuda')


        for i in range(0, placeholder.size(0)):
            dec_input = tgt[:i + 1, :]
            placeholder = placeholder[:i + 1, :]
            out, hidden = self.decoder(dec_input, hidden, placeholder, enc_output)

            pred = out.argmax(dim=-1)
            outputs[i:, :] = out
           

        # outputs(batch,length,vocab_size)
        return outputs.transpose(1, 0)


