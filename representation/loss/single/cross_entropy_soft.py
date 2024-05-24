"""
https://github.com/DingfanChen/RelaxLoss/blob/fc8c68eee62d3cdecadd0c2fcebc47b0b3e56361/source/utils/ops.py#L21
"""

import torch
from torch import nn
import torch.nn.functional as F


def CrossEntropy_soft(input, target, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    logprobs = F.log_softmax(input, dim=1)
    losses = -(target * logprobs)
    if reduction == 'mean':
        return losses.sum() / input.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)
