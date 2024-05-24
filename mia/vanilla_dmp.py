"""
some function are derived from: https://github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/advreg.py
"""
import logging
import os
import time
import copy

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from mia.vanilla import VanillaMia
from representation.aug.aug_selector import get_aug
from conf import settings
from dataloader.info import num_classes

from util.averagemeters import AverageMeter
from util.metrics import accuracy
from util.yaml_loader import load_loss_conf


class VanillaMia_DMP(VanillaMia):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
    ):
        super(VanillaMia_DMP, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
        # models
        self.target_model_up = copy.deepcopy(self.target_model)
        self.shadow_models_up = []
        for m in self.shadow_models:
            self.shadow_models_up += [copy.deepcopy(m)]
        # dataset
        self.target_ref = target_ref
        self.shadow_ref_list = shadow_ref
        # others
        self.producer = None
        self.loss_confs = None
        self.num_class = None
        self.alpha = 0

    def load_components(self, args):
        from representation.aug.aug_selector import get_aug
        self.producer = get_aug(args)
        self.loss_confs = load_loss_conf(args.loss_conf)
        self.num_class = num_classes[args.dataset]
        self.tau = self.loss_confs['tau']

    def train_target_model(self, args):
        save_path = args.save_path = os.path.join(args.save_folder, args.arch, args.arch_conf)
        ''' logger '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args.logger_file = os.path.join(save_path, 'log_{}_target.txt'.format(args.cmd))
        handler = logging.FileHandler(args.logger_file, mode='w')
        formatter = logging.Formatter('%(asctime)s:%(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger('logger_target')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
        """ load aug """
        self.data_aug = get_aug(args=args)
        """ load epoch conf """
        args = self.load_epoch_conf(args, model_type='target', option='train')
        """ put it on multi-gpu """
        model = self.target_model_up.to(args.device)
        ''' others '''
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )

        criterion = nn.CrossEntropyLoss().to(args.device)
        ce = nn.CrossEntropyLoss().to(args.device)
        ''' training up model '''
        best_prec1 = 0
        cur_prec1 = 0
        step = 0
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            self.classification_train(
                args,
                model=model,
                dataloader=self.target_train,
                optimizer=optimizer,
                criterion=criterion,
                current_epoch=current_epoch, logger=logger
            )
            prec1 = self.validate(
                args,
                test_loader=self.target_test, model=model,
                criterion=ce, logger=logger
            )
            cur_prec1 = prec1
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            logger.info(' * best \t{:.3f}\t'.format(best_prec1.item()))
            torch.cuda.empty_cache()
        # save to disk
        final_path = os.path.join(save_path, 'target')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}.path.tar'.format('target_up'))
        torch.save(model.cpu(), final_path)
        ''' selecting ref data '''
        data_x_temp = []
        data_y_temp = []
        model = model.to(args.device).eval()
        for i, (inputs, target) in enumerate(self.target_ref):
            target = target.view(-1).long().to(args.device)
            inputs = inputs.to(args.device)
            out, fea = model(inputs)
            data_x_temp += [inputs.data.cpu().numpy()]
            data_y_temp += [out.data.cpu().numpy()]
        model = model.cpu()
        data_x_temp = np.concatenate(data_x_temp)
        data_y_temp = np.concatenate(data_y_temp)
        ds = RefData(data_x_temp, data_y_temp)
        dler = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
        )
        logger.info(f'***************** size: {data_x_temp.shape}, {data_y_temp.shape}')
        ''' training p model '''
        model = self.target_model.to(args.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        # train
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            self.classification_train_dmp(
                args,
                model=model,
                dataloader=dler,
                optimizer=optimizer,
                current_epoch=current_epoch, logger=logger
            )
            prec1 = self.validate(
                args,
                test_loader=self.target_test, model=model,
                criterion=ce, logger=logger
            )
            cur_prec1 = prec1
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            logger.info(' * best \t{:.3f}\t'.format(best_prec1.item()))
            torch.cuda.empty_cache()
        # save to disk
        final_path = os.path.join(save_path, 'target')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}.path.tar'.format('target'))
        torch.save(model.cpu(), final_path)
        print("-----------------------Target End---------------------")

    def train_shadow_model(self, args, idx):
        save_path = args.save_path = os.path.join(args.save_folder, args.arch, args.arch_conf)
        ''' logger '''
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        args.logger_file = os.path.join(save_path, 'log_{}_shadow_{}.txt'.format(args.cmd, idx))
        handler = logging.FileHandler(args.logger_file, mode='w')
        formatter = logging.Formatter('%(asctime)s:%(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger('logger_shadow_{}'.format(idx))
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)
        logger.addHandler(logging.StreamHandler())
        """ load aug """
        self.data_aug = get_aug(args=args)
        """ load epoch conf """
        args = self.load_epoch_conf(args, model_type='shadow', option='train')
        ''' load model and dataloader '''
        model = self.shadow_models_up[idx].to(args.device)
        train_loader = self.shadow_train_list[idx]
        test_loader = self.shadow_test_list[idx]
        ref_loader = self.shadow_ref_list[idx]
        ''' others '''
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        criterion = nn.CrossEntropyLoss().to(args.device)
        ce = nn.CrossEntropyLoss().to(args.device)
        ''' training '''
        best_prec1 = 0
        cur_prec1 = 0
        step = 0
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            self.classification_train(
                args,
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                current_epoch=current_epoch, logger=logger
            )
            prec1 = self.validate(
                args,
                test_loader=test_loader, model=model,
                criterion=ce, logger=logger
            )
            cur_prec1 = prec1
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            logger.info(' * best \t{:.3f}\t'.format(best_prec1.item()))
            torch.cuda.empty_cache()
        # save to disk
        final_path = os.path.join(save_path, 'shadow')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}_{}.path.tar'.format('shadow_up', idx))
        torch.save(model.cpu(), final_path)
        ''' selecting ref data '''
        data_x_temp = []
        data_y_temp = []
        model = model.to(args.device).eval()
        for i, (inputs, target) in enumerate(ref_loader):
            #target = target.view(-1).long().to(args.device)
            inputs = inputs.to(args.device)
            out, fea = model(inputs)
            data_x_temp += [inputs.data.cpu().numpy()]
            data_y_temp += [out.data.cpu().numpy()]
        model = model.cpu()
        data_x_temp = np.concatenate(data_x_temp)
        data_y_temp = np.concatenate(data_y_temp)
        logger.info(f'***************** size: {data_x_temp.shape}, {data_y_temp.shape}')
        ds = RefData(data_x_temp, data_y_temp)
        dler = torch.utils.data.DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False
        )
        ''' training p model '''
        model = self.shadow_models[idx].to(args.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        # train
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            self.classification_train_dmp(
                args,
                model=model,
                dataloader=dler,
                optimizer=optimizer,
                current_epoch=current_epoch, logger=logger
            )
            prec1 = self.validate(
                args,
                test_loader=test_loader, model=model,
                criterion=ce, logger=logger
            )
            cur_prec1 = prec1
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            logger.info(' * best \t{:.3f}\t'.format(best_prec1.item()))
            torch.cuda.empty_cache()
        # save to disk
        final_path = os.path.join(save_path, 'shadow')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}.path.tar'.format('shadow'))
        torch.save(model.cpu(), final_path)

    def classification_train_dmp(self, args, model, dataloader, optimizer, current_epoch, logger):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        total_losses = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        model.train()

        tau = float(self.tau)
        for i, (inputs, target) in enumerate(dataloader):
            data_time.update(time.time() - end)
            #
            target = target.to(args.device)
            inputs = inputs.to(args.device)
            # data aug
            inputs = self.data_aug(inputs)
            # infer
            output, fea = model(inputs)
            #print(output.size(), target.size())
            loss = F.kl_div(F.log_softmax(output / tau, dim=-1), F.softmax(target / tau, dim=-1), reduction='mean')
            total_loss = loss
            loss.item()
            losses.update(loss.item(), inputs.size(0))
            total_losses.update(total_loss.item(), inputs.size(0))
            #prec1 = accuracy(output.data, target, topk=(1,))
            #top1.update(prec1[0], inputs.size(0))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info("DMP Epoch: [{0}]\t"
                            "Iter: [{1}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t".format(
                            #"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                    current_epoch,
                    i,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=total_losses,
                    #top1=top1
                ))


def accuracy_binary(output, target):
    """Computes the accuracy for binary classification"""
    batch_size = target.size(0)

    pred = output.view(-1) >= 0.5
    truth = target.view(-1) >= 0.5
    acc = pred.eq(truth).float().sum(0).mul_(100.0 / batch_size)
    return acc


class RefData(Dataset):
    def __init__(self, X, Y):
        self.data = X
        self.labels = Y

        # override the length function

    def __len__(self):
        return len(self.data)

    # override the getitem function
    def __getitem__(self, index):
        # load the data at index and apply transform
        data = self.data[index]
        # load the labels into a list and convert to tensors
        labels = self.labels[index]
        # return data labels
        return data, labels


if __name__ == '__main__':
    a = VanillaMia(None, None, None, None, None, None, None)
