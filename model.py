import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pattern_size):
        super(LSTMTagger, self).__init__()

        self.hidden_dim   = hidden_dim
        self.s_embeddings = nn.Embedding(vocab_size, embedding_dim) 
        self.lstm         = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden       = self.init_hidden(1)
        self.hidden2tag   = nn.Linear(2 * embedding_dim + hidden_dim, tagset_size)

        self.lstm2           = nn.LSTM(embedding_dim, hidden_dim)
        self.decoder      = nn.Linear(2 * embedding_dim + hidden_dim + tagset_size ,pattern_size)

        self.tag = None

    def init_hidden(self):
        return (torch.zeros(1, 1, self.hidden_dim),
                torch.zeros(1, 1, self.hidden_dim))

    def forward(self, entity1, entity2, sentence, hidden):
        entities = torch.cat((entity1, entity2), 1)
        embeds = self.s_embeddings(sentence)
        # Classifier
        if not self.tag:
            lstm_out, self.hidden = self.lstm(
                embeds.view(len(sentence), 1, -1), self.hidden)
            out = torch.cat((entities, lstm_out), 1)
            tag_space = self.hidden2tag(out.view(len(sentence), -1))
            tag_scores = F.log_softmax(tag_space, dim=1)
            self.tag = tag_scores
        # Decoder
        lstm_out2, hidden = self.lstm2(
            embeds.view(len(sentence), 1, -1), hidden)
        out2 = torch.cat((entities, lstm_out2), 1)
        pattern = self.decoder(torch.cat((out2.view(len(sentence), -1), tag_space)))

        return pattern, hidden