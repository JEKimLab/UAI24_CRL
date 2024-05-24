import sys
from torch import nn


def get_attack_model(args):
    """ return given model
    """

    if args.attack_method == 'vanilla':
        from mia.models.vanilla.attack_model_selector import get_attack_model
        net = get_attack_model(args)
    elif args.attack_method == 'advdist':
        from mia.models.advdist.attack_model_selector import get_attack_model
        net = get_attack_model(args)
    elif args.attack_method in ['entr', 'mentr']:
        from mia.models.threshold.attack_model_selector import get_attack_model
        net = get_attack_model(args)
    else:
        print('the network name you have entered is not supported in {} yet'.format(args.attack_arch))
        NotImplementedError()
        sys.exit()
    return net
