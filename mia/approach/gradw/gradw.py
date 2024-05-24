import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_curve, accuracy_score
from torch import nn
from torch.utils import data

from mia.approach.gradx.gradx import Gradx
from mia.func.gradient import gradient_based_attack_wrt_x, gradient_based_attack_wrt_w
from mia.func.threshold_base import ThresholdBase
from mia.vanilla import VanillaMia
from conf import settings

from util.averagemeters import AverageMeter
from util.metrics import accuracy
from util.yaml_loader import load_loss_conf

from mia.func.entropy import _entr_comp


class Gradw(Gradx):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(Gradw, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )
        # self.target_ref = target_ref
        # self.shadow_ref = shadow_ref
        self.thres_list = None

    #@torch.no_grad()
    def get_predictions(self, args, model, dataloader, m=10, epsilon=1e-3):
        model = model.to(args.device)
        model.eval()

        entrs, targets = gradient_based_attack_wrt_w(args, dataloader, model)

        torch.cuda.empty_cache()
        model = model.cpu()
        torch.cuda.empty_cache()
        return entrs, targets


if __name__ == '__main__':
    pass
