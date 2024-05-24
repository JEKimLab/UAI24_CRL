"""
derived from: https://github.com/jxgu1016/MNIST_center_loss_pytorch/blob/master/CenterLoss.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LogNormRelaxCenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        fea_dim (int): feature dimension.
    """

    def __init__(self, num_classes, fea_dim, alpha_cl=1.0, upper_cl=1.0, t_cl=1.0, size_average=True, use_gpu=True):
        super(LogNormRelaxCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.fea_dim = fea_dim
        self.alpha = alpha_cl
        self.t = t_cl
        self.upper = upper_cl

        self.size_average = size_average
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.fea_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.fea_dim))
        print(self.centers.size())

    def forward(self, pred, fea, label, epoch):
        batch_size = fea.size(0)
        # normalize
        norms = torch.norm(fea, p=2, dim=-1, keepdim=True) + 1e-7
        #fea_norm = torch.div(fea, norms) / self.t
        fea_norm = torch.div(fea, 1 + norms * self.t)
        norms = torch.norm(self.centers, p=2, dim=-1, keepdim=True) + 1e-7
        #center_norm = torch.div(self.centers, norms) / self.t
        center_norm = torch.div(self.centers, 1 + norms * self.t)
        # To check the dim of centers and features
        bs_tensor = fea_norm.new_empty(1).fill_(batch_size if self.size_average else 1)
        centers_batch = center_norm.index_select(0, label.long())
        dist = (fea_norm - centers_batch).pow(2).sum(dim=1, keepdim=False)
        dist = dist.sum() / bs_tensor / 2.0
        if epoch % 2 == 0:
            loss = ((fea_norm - centers_batch).pow(2).sum() / bs_tensor / 2.0 - self.alpha).abs()
        else:
            if dist > self.alpha:  # normal gradient descent
                loss = dist
            else:
                confidence_target = F.softmax(pred, dim=-1)[torch.arange(label.size(0)), label]
                confidence_target = torch.clamp(confidence_target, min=0., max=self.upper)
                confidence_other = 1 - confidence_target
                loss = (fea_norm - centers_batch).pow(2).sum(dim=1, keepdim=False) * confidence_target \
                       + confidence_other * (fea_norm).pow(2).sum(dim=1, keepdim=False)
                loss = loss.sum() / 2.0 / bs_tensor
        return loss


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
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)


if __name__ == '__main__':
    c = torch.randn(20, 5)
    c = c.repeat(64, 1, 1)
    print(c.size())

