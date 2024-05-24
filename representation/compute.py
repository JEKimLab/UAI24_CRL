import sys
import torch
import torch.nn.functional as F


def compute_similarity(x1, x2, metric='euclid'):
    if metric == 'euclid':
        return euclid_distance(x1, x2)
    elif metric == 'cosine':
        return cosine_similarity(x1, x2)
    else:
        NotImplementedError()
        sys.exit()

def euclid_distance(x1, x2):
    return F.pairwise_distance(x1, x2, p=2)


def cosine_similarity(x1, x2):
    normalized_tensor_1 = x1 / x1.norm(dim=-1, keepdim=True)
    normalized_tensor_2 = x2 / x2.norm(dim=-1, keepdim=True)
    return (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)
