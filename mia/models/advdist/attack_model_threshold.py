import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AttackNetThreshold(nn.Module):
    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, x):
        print(x.size())
        b, _ = x.size()
        out = torch.gt(x, self.threshold).view(-1)
        out = F.one_hot(out.to(torch.int64))
        return out

    def update_threshold(self, threshold):
        if not torch.is_tensor(threshold):
            threshold = torch.tensor(threshold)
        self.threshold = nn.Parameter(threshold)


if __name__ == '__main__':
    print(torch.arange(0, 5) % 3 >= 1)
    t = F.one_hot(torch.arange(0, 5) % 3)
    print(t)
    net = AttackNetThreshold(1.5)
    values = torch.tensor([
        [2], [0], [1]
    ])
    print(net(values))
