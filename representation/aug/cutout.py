import numpy as np
import torch
import torch.nn as nn


class Cutout(nn.Module):
    def __init__(self, n_holes=1, length=8, p=1.0):
        super().__init__()
        self.p = p
        self.cutout = CutoutNP(n_holes=n_holes, length=length)

    def forward(self, x):
        if np.random.random() > self.p:
            return x
        return self.cutout(x)


class CutoutNP(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes=1, length=8):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(-1)
        w = img.size(-2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img).cuda()
        # mask.to(img.device)
        img = img * mask

        return img
