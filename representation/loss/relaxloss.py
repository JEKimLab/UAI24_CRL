import torch
import torch.nn as nn
from torch.nn import functional as F


class RelaxLoss(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(RelaxLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss()
        self.soft_ce = CrossEntropyLoss_soft()
        # hyper-parameters
        self.num_classes = num_classes
        self.alpha = kwargs['alpha']
        self.upper = kwargs['upper']

    def forward(self, pred, label, epoch):
        loss_ce = self.ce(pred, label)
        if epoch % 2 == 0:  # gradient ascent/ normal gradient descent
            loss = (loss_ce - self.alpha).abs()
        else:
            if loss_ce > self.alpha:  # normal gradient descent
                loss = loss_ce
            else:  # posterior flattening
                confidence_target = F.softmax(pred, dim=-1)[torch.arange(label.size(0)), label]
                confidence_target = torch.clamp(confidence_target, min=0., max=self.upper)
                confidence_else = (1.0 - confidence_target) / (self.num_classes - 1)
                onehot = one_hot_embedding(label, num_classes=self.num_classes)
                soft_targets = onehot * confidence_target.unsqueeze(-1).repeat(1, self.num_classes) \
                               + (1 - onehot) * confidence_else.unsqueeze(-1).repeat(1, self.num_classes)
                loss = self.soft_ce(pred, soft_targets)
        return loss


class CrossEntropyLoss_soft(nn.Module):
    """
    derived from: https://github.com/DingfanChen/RelaxLoss/blob/main/source/utils/ops.py
    """

    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss_soft, self).__init__()
        self.reduction = reduction

    def forward(self, pred, label):
        logprobs = F.log_softmax(pred, dim=1)
        losses = -(label * logprobs)
        if self.reduction == 'mean':
            return losses.sum() / pred.shape[0]
        elif self.reduction == 'sum':
            return losses.sum()
        elif self.reduction == 'none':
            return losses.sum(-1)
        else:
            NotImplementedError()


def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    """
    derived from: https://github.com/DingfanChen/RelaxLoss/blob/main/source/utils/ops.py
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    """
    scatter_dim = len(y.size())
    # y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)
