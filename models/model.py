import torch
import torch.nn as nn

class PoemLSTM(nn.Module):
    """
    """

    def __init__(self, hidden_size, vocab_size, embedding_dim=200, n_layers=1):
        super(PoemLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        topic_embedding_dim = 100
        # embedding for topic
        self.embed_topic = nn.Embedding(vocab_size, topic_embedding_dim)  # [41634, 100]

        # embedding for sent
        self.embed_sent = nn.Embedding(vocab_size, embedding_dim)  # [41634, 200]

        # concatenate
        self.lstm = nn.LSTM(topic_embedding_dim + embedding_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)  # [100,41634]

    def forward(self, topic, inp):
        # topic embedding matrix
        topic_emb = self.embed_topic(topic)

        # sent embedding for words
        emb = self.embed_sent(inp)
        #         print('emb out',emb.shape)

        # concatinate input sentence and topic
        input_combined = torch.cat((emb, topic_emb), 1)
        #         print('shape',input_combined.shape)

        # prepare Embedding output for LSTM layer
        lstm_in = input_combined.view(input_combined.shape[0], 1,
                                      input_combined.shape[1])  # lstm_in = emb.view(1, 1, -1)
        #         print('lstm in',lstm_in.shape,'in dim',lstm_in.dim())

        # feed to LSTM layer
        lstm_out, hn_cn = self.lstm(lstm_in)
        #         print('lstm_out shape',lstm_out.size())

        # re-shape LSTM output, prepare for Linear layer
        fc_in = lstm_out.view(len(input_combined), -1)
        #         print('fc in', fc_in.size())

        # feed to Linear
        fc_out = self.fc(fc_in)
        #         print('fc_out',fc_out.shape)

        return fc_out, hn_cn