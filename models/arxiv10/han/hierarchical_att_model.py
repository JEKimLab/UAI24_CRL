"""
from https://github.com/uvipen/Hierarchical-attention-networks-pytorch/blob/master/src/hierarchical_att_model.py
"""
import torch
import torch.nn as nn
from models.arxiv10.han.sent_att_model import SentAttNet
from models.arxiv10.han.word_att_model import WordAttNet


class HierAttNet(nn.Module):
    def __init__(
            self,
            words_dim=300, word_num_hidden=50, sentence_num_hidden=50,
            num_classes=10, pretrained_word2vec_path=''
    ):
        super(HierAttNet, self).__init__()

        self.word_att_net = WordAttNet(pretrained_word2vec_path, words_dim, word_num_hidden)
        self.sent_att_net = SentAttNet(word_num_hidden, sentence_num_hidden, num_classes)

    def forward(self, x):
        x = x.permute(1, 2, 0)  # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = None
        for i in range(num_sentences):
            word_attn = self.word_att_net(x[i, :, :])
            if word_attentions is None:
                word_attentions = word_attn
            else:
                word_attentions = torch.cat((word_attentions, word_attn), 0)
        o, fea = self.sent_att_net(word_attentions)
        return o, fea


def han(
    words_dim=50, word_num_hidden=50, sentence_num_hidden=50,
    pretrained_word2vec_path='', num_classes=10
):
    net = HierAttNet(
        words_dim, word_num_hidden, sentence_num_hidden,
        num_classes, pretrained_word2vec_path
    )
    return net


if __name__ == '__main__':
    net = han(pretrained_word2vec_path='/home/xfang23/dataset/ArXiv-10/glove/glove.6B.50d.txt')
    print(net)
    x = torch.ones(2, 30, 35).long()
    print(torch.softmax(net(x), dim=-1))
