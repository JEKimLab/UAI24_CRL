import torch

def cross_entropy_after_softmax(probs, label, reduction='mean'):
    '''
    cross entropy loss on soft labels
    :param input:
    :param target:
    :param reduction:
    :return:
    '''
    B, C = probs.size()
    logprobs = torch.log(probs)
    onehot = one_hot_embedding(label, num_classes=C)
    losses = -(onehot * logprobs)
    if reduction == 'mean':
        return losses.sum() / label.shape[0]
    elif reduction == 'sum':
        return losses.sum()
    elif reduction == 'none':
        return losses.sum(-1)


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


def l2_softmax(outputs, outputs_center):
    B, C = outputs.size()
    ceters = outputs_center.repeat(B, 1, 1)
    outputs = outputs.view(B, 1, -1)
    dist = (outputs - ceters).pow(2).clamp(min=1e-5).sum(dim=2, keepdim=False)
    probs = torch.softmax(dist, dim=-1)
    return probs
