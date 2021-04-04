import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, text_field, dropout,
                 bidirectional=True):
        super(GRUModel, self).__init__()

        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.predict = False

        self.embedding = nn.Embedding(len(text_field.vocab), input_size)
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=False,
                          bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if self.bidirectional else 1), 1)

    def forward(self, text):
        text_embedded = self.embedding(text)
        # text_embedded shape: [sentence length, batch size, embedding dim]

        # output shape: (seq_len, batch, num_directions * hidden_size)
        # hidden shape: (num_layers * num_directions, batch, hidden_size)
        # num_layers = 1, num_directions = 2 if bidirectional else 1
        output, hidden = self.rnn(text_embedded)

        if self.bidirectional:
            output = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        output = self.fc(output)
        output = self.dropout(output)
        output = torch.squeeze(output, 1)

        return output
