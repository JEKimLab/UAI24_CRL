import sys
from torch import nn
from dataloader.info import num_classes


def get_attack_model(args):
    """ return given model
    """
    dim_input = 1

    if args.attack_arch == 'linear':
        from mia.models.advdist.attack_model_linear import AttackNetLinear
        net = AttackNetLinear(in_dim=dim_input)
    elif args.attack_arch == 'threshold':
        from mia.models.advdist.attack_model_threshold import AttackNetThreshold
        net = AttackNetThreshold()
    else:
        print('the network name you have entered is not supported in {} yet'.format(args.attack_arch))
        NotImplementedError()
        sys.exit()
    return net
