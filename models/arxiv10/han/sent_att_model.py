"""
from https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/src/sent_att_model.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SentAttNet(nn.Module):
    def __init__(self, word_num_hidden=50, sentence_num_hidden=50, num_classes=10):
        super(SentAttNet, self).__init__()
        self.sentence_context_weights = nn.Parameter(torch.rand(2 * sentence_num_hidden, 1))
        self.sentence_context_weights.data.uniform_(-0.1, 0.1)
        self.sentence_gru = nn.GRU(2 * word_num_hidden, sentence_num_hidden, bidirectional=True)
        self.sentence_linear = nn.Linear(2 * sentence_num_hidden, 2 * sentence_num_hidden, bias=True)
        self.fc = nn.Linear(2 * sentence_num_hidden, num_classes)
        self.soft_sent = nn.Softmax()

    def forward(self, x):
        sentence_h, _ = self.sentence_gru(x)
        x = torch.tanh(self.sentence_linear(sentence_h))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_sent(x.transpose(1, 0))
        x = torch.mul(sentence_h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        x = x.squeeze(0)
        fea = x
        x = self.fc(x)
        return x, fea


if __name__ == "__main__":
    abc = SentAttNet()
