import sys

import numpy as np
from torch import nn
from dataloader.info import num_classes


def get_attack_model(args):
    """ return given model
    """
    if args.attack_arch == 'threshold_p_c':
        from mia.models.threshold.attack_model_threshold_pc import AttackNetThresholdPC
        net = AttackNetThresholdPC(threshold=np.zeros((num_classes[args.dataset])))
    else:
        print('the network name you have entered is not supported in {} yet'.format(args.attack_arch))
        NotImplementedError()
        sys.exit()
    return net
