import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_curve, accuracy_score
from torch import nn
from torch.utils import data

from mia.approach.entr.entr import EntrVanilla
from conf import settings

from mia.func.entropy import _m_entr_comp


class MentrVanilla(EntrVanilla):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(MentrVanilla, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )

    @torch.no_grad()
    def get_predictions(self, args, model, dataloader, m=10, epsilon=1e-3):
        model = model.to(args.device)
        model.eval()
        pred_list = None
        target_list = None

        with torch.no_grad():
            for i, (inputs, target) in enumerate(dataloader):
                inputs = inputs.to(args.device)
                pred, fea = model(inputs)
                pred = torch.softmax(pred, dim=-1)
                if pred_list is None:
                    pred_list = pred.detach().cpu().numpy()
                    target_list = target.detach().cpu().numpy()
                else:
                    pred_list = np.concatenate((pred_list, pred.detach().cpu().numpy()), axis=0)
                    target_list = np.concatenate((target_list, target.detach().cpu().numpy()), axis=0)
            entrs = _m_entr_comp(pred_list, target_list)
            torch.cuda.empty_cache()
        targets = target_list
        model = model.cpu()
        torch.cuda.empty_cache()
        return entrs, targets

    @torch.no_grad()
    def get_predictions_test(self, args, model, dataloader):
        model = model.to(args.device)
        model.eval()
        pred_list = None
        target_list = None
        group_list = None

        with torch.no_grad():
            for i, (inputs, target, group_index) in enumerate(dataloader):
                inputs = inputs.to(args.device)
                pred, fea = model(inputs)
                pred = torch.softmax(pred, dim=-1)
                if pred_list is None:
                    pred_list = pred.detach().cpu().numpy()
                    target_list = target.detach().cpu().numpy()
                    group_list = group_index.detach().cpu().numpy()
                else:
                    pred_list = np.concatenate((pred_list, pred.detach().cpu().numpy()), axis=0)
                    target_list = np.concatenate((target_list, target.detach().cpu().numpy()), axis=0)
                    group_list = np.concatenate((group_list, group_index.detach().cpu().numpy()), axis=0)
            entrs = _m_entr_comp(pred_list, target_list)
            torch.cuda.empty_cache()
        targets = target_list
        groups = group_list
        model = model.cpu()
        torch.cuda.empty_cache()
        return entrs, targets, groups


if __name__ == '__main__':
    pass
