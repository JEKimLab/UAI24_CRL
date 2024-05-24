import torch
import torch.nn as nn

from representation.loss.single.relaxloss_lognorm import LogNormRelaxLoss
from representation.loss.single.center_loss_relax_lognorm import LogNormRelaxCenterLoss


class CRL(nn.Module):
    def __init__(self, num_classes, fea_dim, **kwargs):
        super(CRL, self).__init__()
        self.ce = LogNormRelaxLoss(num_classes, **kwargs)
        self.ct = LogNormRelaxCenterLoss(
            num_classes, fea_dim,
            alpha_cl=kwargs['alpha_cl'], t_cl=kwargs['t_cl'], upper_cl=kwargs['upper_cl']
        )
        # hyper-parameters
        self.size_average = True
        self.alpha = kwargs['alpha_cl']
        self.upper = kwargs['upper_cl']
        self.the_lambda = kwargs['lambda']
        #
        self.num_classes = num_classes
        self.fea_dim = fea_dim

    def forward(self, pred, fea, label, epoch):
        return self.ce(pred, label, epoch) + self.the_lambda * self.ct(pred, fea, label, epoch)


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
