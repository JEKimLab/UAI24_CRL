import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class AttackNetThresholdPC(nn.Module):
    def __init__(self, threshold):
        super().__init__()
        self.threshold = nn.Parameter(torch.tensor(threshold))

    def forward(self, x):
        b, _ = x.size()
        x1 = x[:, 0].view(-1)
        x2 = x[:, 1].view(-1)
        threshold = self.threshold.index_select(0, x2.long())
        out = torch.lt(x1, threshold).view(-1)
        out = F.one_hot(out.to(torch.int64))
        return out

    def update_threshold(self, threshold):
        if not torch.is_tensor(threshold):
            threshold = torch.tensor(threshold)
        self.threshold = nn.Parameter(threshold)


if __name__ == '__main__':
    net = AttackNetThresholdPC(np.ones(10))
    values = torch.tensor([
        [10,0], [0,0], [0,1], [0.5,3]
    ])
    print(net(values))
