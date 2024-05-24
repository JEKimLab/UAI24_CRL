import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul(100.0 / batch_size))

    return res


def get_f1_score(output, target, average='weighted'):
    score = f1_score(target, output, average=average)
    return score


def get_precision_score(output, target, average='weighted'):
    score = precision_score(target, output, average=average)
    return score


def get_recall_score(output, target, average='weighted'):
    score = recall_score(target, output, average=average)
    return score


def get_accuracy_score(output, target):
    score = accuracy_score(target, output)
    return score


def get_auc_score(output, target):
    score = roc_auc_score(target, output)
    return score


def transform_one_hot_to_index(x):
    return torch.argmax(x, dim=1).cpu().numpy()


if __name__ == '__main__':
    pass
