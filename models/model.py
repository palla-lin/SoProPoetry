import torch
import torch.nn as nn


class PoemLSTM(nn.Module):

    def __init__(self, hidden_size, vocab_size, sent_embedding_dim, topic_embedding_dim, topic_vocab_size, topic_hidden_size, n_layers=1):
        super(PoemLSTM, self).__init__()

        # embedding for topic
        self.embed_topic = nn.Embedding(topic_vocab_size, topic_embedding_dim)

        # embedding for sent
        self.embed_sent = nn.Embedding(vocab_size, sent_embedding_dim)

        # topic LSTM
        self.topic_lstm = nn.LSTM(topic_embedding_dim, topic_hidden_size)

        # sent LSTM
        self.sent_lstm = nn.LSTM(sent_embedding_dim + topic_hidden_size, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, topic, inp):

        # topic embedding
        topic_emb = self.embed_topic(topic)
        top_lstm_out, hn_cn_top = self.topic_lstm(topic_emb.view(len(topic), 1, topic_emb.shape[1]))

        # sent embedding
        sent_emb = self.embed_sent(inp)
        reshape1 = top_lstm_out.view(top_lstm_out.shape[0], top_lstm_out.shape[2])

        # concatenate input sentence and topic
        input_combined = torch.cat((sent_emb, reshape1), 1)
        #         print('shape',input_combined.shape)

        # pass to word LSTM layer
        sent_lstm_output, sent_Hn_Cn = self.sent_lstm(input_combined.view(len(inp), 1, input_combined.shape[1]))

        # rehsape to pass to Linear layer
        lstm_output = sent_lstm_output.view(sent_lstm_output.shape[0],
                                            sent_lstm_output.shape[2])

        # project the LSTM to linear
        fc_out = self.fc(lstm_output)

        return fc_out