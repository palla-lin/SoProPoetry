# -*- coding: utf-8 -*-
# Peilu
# December 2021

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout):
        super(EncoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, bidirectional=False)

    def forward(self, x):
        # x(length,batch_size)
        # embedded(length,batch_size,hidden_size)
        embedded = self.dropout(self.embedding(x.transpose(1, 0)))
        self.gru.flatten_parameters()
        out, hidden = self.gru(embedded)
        # hidden = hidden[-1] + hidden[-2]
        # out = out[:, :, :self.hidden_size] + out[:, :, self.hidden_size:]
        # print(embedded.shape,out.shape,hidden.shape)
        # exit()
        # out(length, batch_size, hidden_size)
        # hidden(batch_size, hidden_size)
        return out, hidden


class AttDecoderRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout_rate):
        super(AttDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.attn = Attention(hidden_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size * 2, hidden_size)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers=num_layers)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, tgt, hidden, placeholder, enc_output):
        # tgt(batch, 1)
        # placeholder(batch_size, length, hidden_size)
        # hidden(num_layers, batch_size,hidden_size)
        # enc_output(length, batch_size, hidden_size)
        tgt = tgt.transpose(1, 0)
        placeholder = placeholder.transpose(1, 0)
        # tgt_emb(1, batch_size, hidden_size)
        tgt_emb = self.embedding(tgt) + self.embedding(placeholder)
        tgt_emb_dropout = self.dropout(tgt_emb)
        # attn(batch_size, 1, length)
        # c(1, batch_size, hidden_size)
        # input(1, batch_size, hidden_size*2)
        attn = self.attn(hidden, enc_output).unsqueeze(1)
        c = torch.bmm(attn, enc_output.transpose(1, 0)).transpose(0, 1)
        input = torch.cat((tgt_emb_dropout, c), dim=2)

        self.gru.flatten_parameters()
        output, hidden = self.gru(input, hidden)
        # print('\ntired',output.shape)

        output = self.linear2(output)
        # print('\ntired', output.shape)
        # exit()
        return output, hidden

    def attention(self, h, s):
        # s(number_layers,batch,hidden_size)
        # h(length,batch,hidden_size)
        s = s[-1:].transpose(2, 1)  # s(1,batch,hidden_size)
        weight = F.softmax(torch.matmul(h, s), dim=0)  # weight(length,batch,batch)
        weighted_h = torch.matmul(weight, h)  # weighted_h(length,batch,hidden_size)
        context = torch.sum(weighted_h, 0, keepdim=True)  # context(1,batch,hidden_size)
        return context.transpose(1, 0)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim, bias=False)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, s, enc_output):
        s = s[-1]  # only the last layer
        # s = [batch_size, dec_hid_dim]
        # enc_output = [src_len, batch_size, enc_hid_dim * 2]
        batch_size = enc_output.shape[1]
        src_len = enc_output.shape[0]
        # repeat decoder hidden state src_len times
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
    def __init__(self, vocab_size, hidden_size, num_layers=2, dropout=0.1):
        super(Seq2seq, self).__init__()
        self.vocab_size = vocab_size
        self.encoder = EncoderRNN(vocab_size, hidden_size, num_layers, dropout)
        self.decoder = AttDecoderRNN(vocab_size, hidden_size, num_layers, dropout)

    def forward(self, src, tgt, placeholder):
        # src(batch_size,keyword_length)
        # tgt(batch_size,<bos>+body_length+<eos>)
        # placeholder(batch_size,body_length)

        enc_output, hidden = self.encoder(src)  # hidden will be the initial hidden state of the decoder
        outputs = torch.zeros(placeholder.size(0), placeholder.size(1), self.vocab_size).to('cuda')
        for i in range(0, placeholder.size(1)):
            dec_input = tgt[:, i].unsqueeze(1)
            out, hidden = self.decoder(dec_input, hidden, placeholder[:, i].unsqueeze(1), enc_output)

            outputs[:, i:] = out.transpose(1, 0)
        # outputs(batch,length,vocab_size)
        return outputs

#
