import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split


# import skorch
# from mia.estimators import ShadowModelBundle, AttackModelBundle, prepare_attack_data


class AttackNetLinear(nn.Module):
    def __init__(self, in_dim=100):
        super().__init__()
        self.clf = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.clf(x)
        #return torch.softmax(x, dim=-1)
        return x

if __name__ == '__main__':
    x = torch.tensor([[i for i in range(100)],
                      [i for i in range(100)]],
                     dtype=torch.float32)
    print(x.size())
    model = AttackNetLinear()
    print(model(x))
