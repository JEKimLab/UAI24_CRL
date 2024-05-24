import torch.nn as nn


class DistillationLoss(nn.Module):
    """
    the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities
    """

    def __init__(self, alpha, temperature, criterion, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.T = temperature
        self.reduction = reduction
        self.criterion = criterion
        self.softmax = nn.Softmax(dim=-1)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.KL = nn.KLDivLoss(reduction=reduction)

    def forward(self, preds, target, teacher_outputs):
        loss = self.criterion(preds, target)
        KL = self.KL(self.logsoftmax(preds / self.T), self.softmax(teacher_outputs / self.T))
        KD_loss = KL * (self.alpha * self.T * self.T) + loss * (1. - self.alpha)
        return KD_loss
