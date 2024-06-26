"""
Derived from: https://github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/label_smoothing.py
"""
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingCrossEntropy(nn.Module):
    """
    epsilon * 1/K \sum_{i=1}^K -\log p(yi|x) + (1-epsilon)* \sum_{i=1}^K -ti \log p(yi|x)
    =  epsilon * KL (u || p(y|x)) + const + (1-epsilon)* CE(p(y|x), target)
    """

    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = self.reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return self.linear_combination(loss / n, nll, self.epsilon)

    @staticmethod
    def linear_combination(x, y, epsilon):
        return epsilon * x + (1 - epsilon) * y

    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss
