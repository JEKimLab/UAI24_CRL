''' VGG in PyTorch
[1] Simonyan, Karen, and Andrew Zisserman.
    Very deep convolutional networks for large-scale image recognition
    ICLR
    Modified from https://github.com/chengyangfu/pytorch-vgg-cifar10/blob/master/vgg.py
'''
import math
import sys

import torch.nn as nn

fc_in = 512


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features, num_classes=100):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(fc_in, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
        )
        self.final = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        fea = x
        x = self.final(x)
        return x, fea

    def forward_last(self, fea):
        return self.final(fea)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.GroupNorm(num_groups=32, num_channels=v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    global fc_in
    fc_in = in_channels
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11(num_classes):
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), num_classes)


def vgg11_bn(num_classes):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True), num_classes)


def vgg13(num_classes):
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), num_classes)


def vgg13_bn(num_classes):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True), num_classes)


def vgg16(num_classes):
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), num_classes)


def vgg16_bn(num_classes):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True), num_classes)


def vgg19(num_classes):
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), num_classes)


def vgg19_bn(num_classes):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True), num_classes)


def vgg(num_classes, arch='vgg16'):
    if arch == 'vgg11':
        return vgg11(num_classes)
    elif arch == 'vgg11_bn':
        return vgg11_bn(num_classes)
    elif arch == 'vgg13':
        return vgg13(num_classes)
    elif arch == 'vgg13_bn':
        return vgg13_bn(num_classes)
    elif arch == 'vgg16':
        return vgg16(num_classes)
    elif arch == 'vgg16_bn':
        return vgg16_bn(num_classes)
    elif arch == 'vgg19':
        return vgg19(num_classes)
    elif arch == 'vgg19_bn':
        return vgg19_bn(num_classes)
    else:
        print_not_support(arch)


def print_not_support(arch):
    """ return unsupported info
    """
    print('the network {} you have entered is not supported yet'.format(arch))
    sys.exit()
