"""
from: https://github.com/castorini/hedwig/blob/master/models/han/word_level_rnn.py
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import csv


class WordAttNet(nn.Module):
    def __init__(self, word2vec_path, words_dim=300, word_num_hidden=50):
        super(WordAttNet, self).__init__()
        dict = pd.read_csv(filepath_or_buffer=word2vec_path, header=None, sep=" ", quoting=csv.QUOTE_NONE).values[:, 1:]
        dict_len, embed_size = dict.shape
        dict_len += 1
        unknown_word = np.zeros((1, embed_size))
        dict = torch.from_numpy(np.concatenate([unknown_word, dict], axis=0).astype(np.float32))

        self.word_context_weights = nn.Parameter(torch.rand(2 * word_num_hidden, 1))
        self.GRU = nn.GRU(words_dim, word_num_hidden, bidirectional=True)
        self.linear = nn.Linear(2 * word_num_hidden, 2 * word_num_hidden, bias=True)
        self.word_context_weights.data.uniform_(-0.25, 0.25)
        self.soft_word = nn.Softmax()
        self.lookup = nn.Embedding(num_embeddings=dict_len, embedding_dim=embed_size).from_pretrained(dict)

    def forward(self, x):
        x = self.lookup(x).float()
        h, _ = self.GRU(x)
        x = torch.tanh(self.linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = self.soft_word(x.transpose(1, 0))
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        return x


if __name__ == "__main__":
    abc = WordAttNet("../data/glove.6B.50d.txt")
