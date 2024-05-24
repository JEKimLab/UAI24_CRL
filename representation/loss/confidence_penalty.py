"""
Derived from: https://github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/confidence_penalty.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidencePenalty(nn.Module):
    def __init__(self, alpha: float = 0.1, reduction='mean'):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.reduction = reduction
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, preds, target):
        loss = self.criterion(preds, target)
        probs = self.softmax(preds)
        logprobs = self.logsoftmax(preds)
        entropy = self.reduce_loss(torch.mul(probs, logprobs).sum(dim=-1), self.reduction)  # = negated entropy
        return loss + self.alpha * entropy

    @staticmethod
    def reduce_loss(loss, reduction='mean'):
        return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss
