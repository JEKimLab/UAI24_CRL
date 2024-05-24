import torch
from torch import nn

class Linear1(nn.Module):
    def __init__(self, in_dim=100, num_classes=10, scale=8):
        super().__init__()
        hidden_dim = int(in_dim*scale)
        self.hidden_layers = nn.Linear(in_dim, hidden_dim)
        self.clf = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.hidden_layers(x)
        fea = x
        x = self.clf(x)
        return x, fea