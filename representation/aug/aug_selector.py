import torch
import torchvision.transforms as T

from dataloader import info
from representation.aug.cutout import Cutout
from representation.aug.gaussian_blur import GaussianBlur


def get_aug(args):
    if args.dataset == "cifar100":
        func = torch.nn.Sequential(
            T.RandomCrop(32, 4),
            T.RandomHorizontalFlip(),
        )
    elif args.dataset == "tinyimagenet":
        func = torch.nn.Sequential(
            Cutout(n_holes=1, length=8, p=1.0),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
        )
    else:
        from representation.aug.nothing import Nothing
        func = Nothing()
    return func
