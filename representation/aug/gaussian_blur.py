import random

from PIL import ImageFilter


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=None):
        if sigma is None:
            sigma = [.1, 2.]
        self.sigma = sigma

    def __call__(self, x):
        x_1 = x
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x_2 = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x_1, x_2
