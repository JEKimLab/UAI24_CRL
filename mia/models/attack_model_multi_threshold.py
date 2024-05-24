import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AttackNetMultiThreshold(nn.Module):
    def __init__(self, num_classes=10, threshold=None):
        super().__init__()
        if threshold is None:
            threshold = [0.0 for _ in range(num_classes)]
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, x):
        b, _ = x.size()
        out = torch.gt(x, self.threshold).view(-1)
        out = F.one_hot(out.to(torch.int64))
        return out

    def update_threshold(self, threshold):
        for i in range(len(threshold)):
            if not torch.is_tensor(threshold[i]):
                threshold[i] = torch.tensor(threshold[i])
        self.threshold = nn.Parameter(threshold)


if __name__ == '__main__':
    print(torch.arange(0, 5) % 3 >= 1)
    t = F.one_hot(torch.arange(0, 5) % 3)
    print(t)
    net = AttackNetMultiThreshold(threshold=[1.5 for i in range(10)])
    values = torch.tensor([
        [2], [0], [1]
    ])
    print(net(values))
