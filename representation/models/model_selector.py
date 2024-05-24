import sys

from representation.models.linear1 import Linear1

hidden_dim = {
    'vgg': 512,
    'resnet': 1,
}

def get_network(args):
    """ return given network
    """
    num_classes = args.dataset
    in_dim = hidden_dim[args.arch]
    net = Linear1(in_dim=in_dim, num_classes=num_classes, scale=8)
    #else:
    #    print_not_support(args.dataset)
    return net


#def print_not_support(dataset):
#    """ return unsupported info
#    """
#    print('the network name you have entered is not supported in {} yet'.format(dataset))
#    sys.exit()