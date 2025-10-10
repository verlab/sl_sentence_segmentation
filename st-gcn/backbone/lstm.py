# Code adapted from the code on https://github.com/hannahbull/sign_language_segmentation/blob/master/utils/lstm.py

import torch
from torch import nn
import torch.nn.functional as F

class SimpleLSTM(nn.Module):
    """
    Simple LSTM model based on:
    https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
    """
    def __init__(self, emb_dim=256, hidden_size=64, num_layers=1, bidirectional=False):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.hidden_dim = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            emb_dim, hidden_size, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True
        ).to(self.device)
        if self.bidirectional:
            self.hidden2tag = nn.Linear(2*hidden_size, 1).to(self.device)
        else:
            self.hidden2tag = nn.Linear(hidden_size, 1).to(self.device)
        self.hidden = None

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return (
                torch.zeros(2*self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(2*self.num_layers, batch_size, self.hidden_dim)
            )
        else:
            return (
                torch.zeros(1*self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(1*self.num_layers, batch_size, self.hidden_dim),
            )

    def forward(self, embeds):
        self.hidden = self.init_hidden(embeds.shape[0])
        self.hidden = (self.hidden[0].type(torch.FloatTensor).to(self.device),
                       self.hidden[1].type(torch.FloatTensor).to(self.device))
        x, self.hidden = self.lstm(embeds, self.hidden)
        x = self.hidden2tag(x)
        x = x.view(embeds.shape[0], -1)
        return x
