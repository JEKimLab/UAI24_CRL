import random

from PIL import ImageFilter

class Nothing(object):
    def __init__(self):
        pass

    def __call__(self, x):
        return x