import logging
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_curve, accuracy_score
from torch import nn
from torch.utils import data

from mia.func.threshold_base import ThresholdBase
from mia.vanilla import VanillaMia
from conf import settings

from util.averagemeters import AverageMeter
from util.metrics import accuracy
from util.yaml_loader import load_loss_conf

from mia.func.entropy import _entr_comp


class EntrVanilla(VanillaMia):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(EntrVanilla, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )
        # self.target_ref = target_ref
        # self.shadow_ref = shadow_ref
        self.thres_list = None

    def train_attack_model(self, args):
        if self.attack_train is None:
            self.produce_attack(args)
        self.train_attck_threshold_model(args)

    def train_attck_threshold_model(self, args):
        save_path = args.save_path = os.path.join(args.save_folder, args.arch, args.arch_conf)
        ''' logger '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args.logger_file = os.path.join(save_path, 'log_{}_attack.txt'.format(args.cmd))
        handler = logging.FileHandler(args.logger_file, mode='w')
        formatter = logging.Formatter('%(asctime)s:%(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger('logger_attack')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
        """ load epoch conf """
        args = self.load_epoch_conf(args, model_type='attack', option='train')
        model = self.attack_model.to(args.device)
        ''' load model and dataloader '''
        train_loader = self.attack_train
        test_loader = self.attack_test
        ''' train attack model '''
        model.update_threshold(self.thres_list)
        ''' eval '''
        prec1 = self.validate_threshold(args, test_loader, model, logger)
        logger.info(' * best \t{:.3f}\t'.format(prec1.item()))
        torch.cuda.empty_cache()
        # save to disk
        final_path = os.path.join(save_path, 'attack')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}.path.tar'.format('attack'))
        torch.save(model.cpu(), final_path)

    def validate_threshold(self, args, test_loader, model, logger):
        top1 = AverageMeter()
        tr_values = []
        te_values = []
        for i, (inputs, target, index_class) in enumerate(test_loader):
            for idx in range(inputs.size(0)):
                if target[idx] == 1:
                    tr_values += [inputs[idx]]
                else:
                    te_values += [inputs[idx]]
        tr_values = torch.stack(tuple(tr_values), dim=0)
        te_values = torch.stack(tuple(te_values), dim=0)
        if len(tr_values.size()) == 1:
            tr_values = torch.unsqueeze(tr_values, 1)
        if len(te_values.size()) == 1:
            te_values = torch.unsqueeze(te_values, 1)
        tr_output = model(tr_values)
        te_output = model(te_values)
        # eval
        prec1 = accuracy(tr_output.data, torch.ones(tr_output.size(0)), topk=(1,))
        top1.update(prec1[0], tr_output.size(0))
        prec1 = accuracy(te_output.data, torch.zeros(te_output.size(0)), topk=(1,))
        top1.update(prec1[0], te_output.size(0))
        logger.info("Loss {loss:.3f}\t"
                    "Prec@1 {top1.avg:.3f}\t".format(
            loss=0,
            top1=top1))
        logger.info(' * Prec@1 \t{top1.avg:.3f}\t'.format(top1=top1))
        return top1.avg

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
            entrs = _entr_comp(pred_list)
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
            entrs = _entr_comp(pred_list)
            torch.cuda.empty_cache()
        targets = target_list
        groups = group_list
        model = model.cpu()
        torch.cuda.empty_cache()
        return entrs, targets, groups

    @torch.no_grad()
    def produce_attack(self, args):
        """ make attack dataset """
        target_model = self.target_model.cpu()
        for i in range(len(self.shadow_models)):
            self.shadow_models[i] = self.shadow_models[i].cpu()
        torch.cuda.empty_cache()
        ''' attack training data '''
        data_shadow_train_list = None
        data_shadow_test_list = None
        data_shadow_train_class_list = None
        data_shadow_test_class_list = None
        for i in range(len(self.shadow_models)):
            cur_shadow_model = self.shadow_models[i]
            cur_train_loader = self.shadow_train_list[i]
            cur_test_loader = self.shadow_test_list[i]
            preds, targets = self.get_predictions(args, cur_shadow_model, cur_train_loader)
            if data_shadow_train_list is None:
                data_shadow_train_list = preds
                data_shadow_train_class_list = targets
            else:
                data_shadow_train_list = np.concatenate((data_shadow_train_list, preds), axis=0)
                data_shadow_train_class_list = np.concatenate((data_shadow_train_class_list, targets), axis=0)
            preds, targets = self.get_predictions(args, cur_shadow_model, cur_test_loader)
            if data_shadow_test_list is None:
                data_shadow_test_list = preds
                data_shadow_test_class_list = targets
            else:
                data_shadow_test_list = np.concatenate((data_shadow_test_list, preds), axis=0)
                data_shadow_test_class_list = np.concatenate((data_shadow_test_class_list, targets), axis=0)
        ''' attack testing data '''
        data_target_train_list = None
        data_target_test_list = None
        data_target_train_class_list = None
        data_target_test_class_list = None
        preds, targets = self.get_predictions(args, target_model, self.target_train)
        data_target_train_list = preds
        data_target_train_class_list = targets
        preds, targets = self.get_predictions(args, target_model, self.target_test)
        data_target_test_list = preds
        data_target_test_class_list = targets

        ''' set threshold tool '''
        from dataloader.info import num_classes
        from dataloader.attack.data_attack_entr import MIADataset

        tool = ThresholdBase(
            data_shadow_train_list, data_shadow_test_list, data_target_train_list, data_target_test_list
        )
        tool.load_labels(
            data_shadow_train_class_list, data_shadow_test_class_list,
            data_target_train_class_list, data_target_test_class_list,
            num_classes[args.dataset]
        )
        self.thres_list = tool._mem_inf_thre_perclass()

        ''' load dataloader '''
        attack_train_dataset = MIADataset(
            data_shadow_train_list, data_shadow_test_list,
            data_shadow_train_class_list, data_shadow_test_class_list,
            num_classes=num_classes[args.dataset]
        )
        attack_test_dataset = MIADataset(
            data_target_train_list, data_target_test_list,
            data_target_train_class_list, data_target_test_class_list,
            num_classes=num_classes[args.dataset]
        )
        self.attack_train = data.DataLoader(
            attack_train_dataset,
            batch_size=args.attack_batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=False
        )
        self.attack_test = data.DataLoader(
            attack_test_dataset,
            batch_size=args.attack_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
        )

    @torch.no_grad()
    def produce_attack_test(self, args):
        """ make attack dataset """
        target_model = self.target_model.cpu()
        for i in range(len(self.shadow_models)):
            self.shadow_models[i] = self.shadow_models[i].cpu()
        torch.cuda.empty_cache()
        #''' attack training data '''
        #data_shadow_train_list = None
        #data_shadow_test_list = None
        #data_shadow_train_class_list = None
        #data_shadow_test_class_list = None
        #for i in range(len(self.shadow_models)):
        #    cur_shadow_model = self.shadow_models[i]
        #    cur_train_loader = self.shadow_train_list[i]
        #    cur_test_loader = self.shadow_test_list[i]
        #    preds, targets = self.get_predictions(args, cur_shadow_model, cur_train_loader)
        #    if data_shadow_train_list is None:
        #        data_shadow_train_list = preds
        #        data_shadow_train_class_list = targets
        #    else:
        #        data_shadow_train_list = np.concatenate((data_shadow_train_list, preds), axis=0)
        #        data_shadow_train_class_list = np.concatenate((data_shadow_train_class_list, targets), axis=0)
        #    preds, targets = self.get_predictions(args, cur_shadow_model, cur_test_loader)
        #    if data_shadow_test_list is None:
        #        data_shadow_test_list = preds
        #        data_shadow_test_class_list = targets
        #    else:
        #        data_shadow_test_list = np.concatenate((data_shadow_test_list, preds), axis=0)
        #        data_shadow_test_class_list = np.concatenate((data_shadow_test_class_list, targets), axis=0)
        ''' attack testing data '''
        data_target_train_list = None
        data_target_test_list = None
        data_target_train_class_list = None
        data_target_test_class_list = None
        data_target_train_group_list = None
        data_target_test_group_list = None
        preds, targets = self.get_predictions(args, target_model, self.target_train)
        data_target_train_list = preds
        data_target_train_class_list = targets
        #data_target_train_group_list = groups
        preds, targets = self.get_predictions(args, target_model, self.target_test)
        data_target_test_list = preds
        data_target_test_class_list = targets
        #data_target_test_group_list = groups

        from dataloader.info import num_classes
        from dataloader.attack.data_attack_entr import MIADataset
        ''' load dataloader '''
        attack_test_dataset = MIADataset(
            data_target_train_list, data_target_test_list,
            data_target_train_class_list, data_target_test_class_list,
            num_classes=num_classes[args.dataset],
            #data_train_group_list=None,
            #data_test_group_list=None
        )
        self.attack_test = data.DataLoader(
            attack_test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
        )


if __name__ == '__main__':
    pass
