import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderDecoder(nn.Module):

    def __init__(self, config):
        super(EncoderDecoder, self).__init__()

        self.config = config
        encoder_final_size = 2*config["hidden_size"] if config["bidirectional"] else config["hidden_size"]

        self.encoder = Encoder(config["embed_size"], 
                               config["hidden_size"], 
                               config["bidirectional"], 
                               config["num_layers"], 
                               config["dropout"])

        self.decoder = Decoder(config["embed_size"],
                               config["hidden_size"],
                               encoder_final_size,
                               config["add_attention"],
                               config["num_layers"],
                               config["dropout"],
                               config["max_output_len"])

        self.embed = nn.Embedding(config["vocab_size"], 
                                  config["embed_size"], 
                                  padding_idx=3)

        self.bridge = nn.Linear(2*config["hidden_size"], config["hidden_size"], bias=True)

        self.generator = Generator(config["hidden_size"], 
                                   config["vocab_size"])

    def forward(self, context, target_seq):
        encoder_hidden, encoder_final = self.encode(context)

        target_seq_embed = self.embed(target_seq)
        
        decoder_hidden = self.bridge(encoder_final) if self.config["bidirectional"] else encoder_final
        decoder_hidden = torch.tanh(decoder_hidden)

        decoder_states = []
        pre_output_vectors = []

        for t in range(self.config["poem_size"]):
            decoder_input = target_seq_embed[t,:].unsqueeze(1)
            _, decoder_hidden, pre_output = self.decoder(decoder_input, encoder_hidden, decoder_hidden)
            decoder_states.append(decoder_hidden)
            pre_output_vectors.append(pre_output)

        decoder_states = torch.cat(decoder_states, dim=1)
        pre_output_vectors = torch.cat(pre_output_vectors, dim=1)

        return decoder_states, decoder_hidden, pre_output_vectors

    def encode(self, context):
        return self.encoder(self.embed(context).transpose(0, 1))

    def decode(self, prev_token, encoder_hidden, decoder_hidden):
        return self.decoder(self.embed(prev_token), encoder_hidden, decoder_hidden)

class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, bidirectional, num_layers, dropout):
        super(Encoder, self).__init__()

        self.num_layers = int(num_layers)
        self.bidirectional = bidirectional
        self.rnn = nn.GRU(input_size, hidden_size, self.num_layers, batch_first=True,
                          bidirectional=self.bidirectional, dropout=dropout)

    def forward(self, context):
        output, final = self.rnn(context)

        if self.bidirectional:
            fwd_final = final[0:final.size(0):2]
            bwd_final = final[1:final.size(0):2]
            final = torch.cat([fwd_final, bwd_final], dim=2)  # [num_layers, batch, 2*dim]

        return output, final


class Decoder(nn.Module):

    def __init__(self, embed_size, 
                       hidden_size, 
                       encoder_final_size, 
                       add_attention, 
                       num_layers, 
                       dropout, 
                       max_output_len):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.add_attention = add_attention
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_output_len = max_output_len

        if self.add_attention:
            # Code to add attention
            pass

        self.rnn = nn.GRU(embed_size+encoder_final_size, hidden_size, 
                          self.num_layers, batch_first=True, dropout=self.dropout)

        self.dropout_layer = nn.Dropout(p=dropout)
        self.pre_output_layer = nn.Linear(hidden_size + encoder_final_size + embed_size,
                                          hidden_size, bias=False)

    def forward(self, prev_token, encoder_output, decoder_hidden):
        # Simple summation over context embeddings, should consider attention
        encoder_output = torch.sum(encoder_output, dim=1, keepdim=True)

        # update rnn hidden state
        rnn_input = torch.cat([prev_token, encoder_output], dim=2)
        output, hidden = self.rnn(rnn_input, decoder_hidden)

        pre_output = torch.cat([prev_token, output, encoder_output], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        return output.transpose(0, 1), hidden, pre_output


class Generator(nn.Module):

    def __init__(self, hidden_size, vocab_size):
        super(Generator, self).__init__()

        self.project = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        return F.log_softmax(self.project(x), dim=-1)
