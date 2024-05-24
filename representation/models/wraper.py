import torch
from torch import nn

from representation.models.model_selector import get_network


#from representation.models.model_selector

class Wraper(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.model = model
        self.ext = get_network(args)

    def forward(self, x):
        y1, fea1 = self.model(x)
        y2, fea2 = self.ext(fea1)
        return y1, y2, fea1, fea2

