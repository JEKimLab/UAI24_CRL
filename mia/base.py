import os
import sys

import yaml
import time
import logging

import torch
from torch import nn
import torch.utils.data as data
from torch.functional import F
import torchvision.transforms as T

from conf import settings
from representation.aug.aug_selector import get_aug
from util.averagemeters import AverageMeter
from util.metrics import accuracy


class BaseMia:
    def __init__(self,
                 target_model,
                 shadow_models,
                 attack_model,
                 ):
        super().__init__()
        # model
        self.target_model = target_model
        self.shadow_models = shadow_models
        self.attack_model = attack_model
        # dataset
        self.target_test = None
        self.target_train = None
        self.shadow_test_list = None
        self.shadow_train_list = None
        # attack dataset
        self.attack_train = None
        self.attack_test = None
        # data aug
        self.data_aug = None

    def load_dataset(self, target_train, target_test, shadow_train_list, shadow_test_list):
        self.target_train = target_train
        self.target_test = target_test
        self.shadow_train_list = shadow_train_list
        self.shadow_test_list = shadow_test_list


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
        if args.device == 'cuda':
            #print("---------------------------------CUDA-------------------------------------")
            model = nn.DataParallel(self.target_model).to(args.device)
            print("is p:", isinstance(model, nn.DataParallel))
        else:
            model = self.target_model.to(args.device)
        ''' others '''
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        ''' training '''
        best_prec1 = 0
        cur_prec1 = 0
        step = 0
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            self.classification_train(
                args,
                model=model, dataloader=self.target_train,
                optimizer=optimizer, criterion=criterion,
                current_epoch=current_epoch, logger=logger
            )
            prec1 = self.validate(
                args,
                test_loader=self.target_test, model=model,
                criterion=criterion, logger=logger
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
        if args.device == 'cuda':
            torch.save(model.module.cpu(), final_path)
        else:
            torch.save(model.cpu(), final_path)

    def train_shadow_models(self, args):
        for i in range(len(self.shadow_models)):
            self.train_shadow_model(args, i)

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
        if args.device == 'cuda':
            #print("---------------------------------CUDA-------------------------------------")
            model = nn.DataParallel(self.shadow_models[idx]).to(args.device)
        else:
            model = self.shadow_models[idx].to(args.device)
        train_loader = self.shadow_train_list[idx]
        test_loader = self.shadow_test_list[idx]
        ''' others '''
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        ''' training '''
        best_prec1 = 0
        cur_prec1 = 0
        step = 0
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            self.classification_train(
                args,
                model=model, dataloader=train_loader,
                optimizer=optimizer, criterion=criterion,
                current_epoch=current_epoch, logger=logger
            )
            prec1 = self.validate(
                args,
                test_loader=test_loader, model=model,
                criterion=criterion, logger=logger
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
        final_path = os.path.join(final_path, 'model_latest_{}_{}.path.tar'.format('shadow', idx))
        if args.device == 'cuda':
            torch.save(model.module.cpu(), final_path)
        else:
            torch.save(model.cpu(), final_path)

    def classification_train(self, args, model, dataloader, optimizer, criterion, current_epoch, logger):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        total_losses = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        model.train()
        for i, (inputs, target) in enumerate(dataloader):
            data_time.update(time.time() - end)
            #
            target = target.view(-1).long().to(args.device)
            inputs = inputs.to(args.device)
            # data aug
            inputs = self.data_aug(inputs)
            # infer
            output, fea = model(inputs)
            loss = criterion(output, target)
            total_loss = loss
            loss.item()
            losses.update(loss.item(), inputs.size(0))
            total_losses.update(total_loss.item(), inputs.size(0))
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], inputs.size(0))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info("Epoch: [{0}]\t"
                            "Iter: [{1}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                    current_epoch,
                    i,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=total_losses,
                    top1=top1)
                )

    def attack_classification_train(self, args, model, dataloader, optimizer, criterion, current_epoch, logger):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        total_losses = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()
        for i, (inputs, target, index_class) in enumerate(dataloader):
            if len(inputs.size()) == 1:
                inputs = torch.unsqueeze(inputs, 1)
            #print(inputs.size())
            data_time.update(time.time() - end)
            target = target.view(-1).view(-1).to(args.device)
            inputs = inputs.to(args.device)
            output = model(inputs)
            #print(output.size(), target.size())
            #print(target)
            loss = criterion(output, target)
            total_loss = loss
            loss.item()
            losses.update(loss.item(), inputs.size(0))
            total_losses.update(total_loss.item(), inputs.size(0))
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], inputs.size(0))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            batch_time.update(time.time() - end)
            end = time.time()
            if i % args.print_freq == 0:
                logger.info("Epoch: [{0}]\t"
                            "Iter: [{1}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                    current_epoch,
                    i,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=total_losses,
                    top1=top1)
                )

    def train_attack_model(self, args):
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
        ''' load model and dataloader '''
        model = self.attack_model.to(args.device)
        train_loader = self.attack_train
        test_loader = self.attack_test
        ''' others '''
        criterion = nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        ''' training '''
        best_prec1 = 0
        cur_prec1 = 0
        step = 0
        for current_epoch in range(args.start_epoch, args.epoch):
            model.train()
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            self.attack_classification_train(
                args,
                model=model, dataloader=train_loader,
                optimizer=optimizer, criterion=criterion,
                current_epoch=current_epoch, logger=logger
            )
            prec1 = self.validate_attack(
                args,
                test_loader=test_loader, model=model,
                criterion=criterion, logger=logger
            )
            cur_prec1 = prec1
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            logger.info(' * best \t{:.3f}\t'.format(best_prec1.item()))
            torch.cuda.empty_cache()
        # save to disk
        final_path = os.path.join(save_path, 'attack')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}.path.tar'.format('attack'))
        torch.save(model.cpu(), final_path)

    def adjust_learning_rate(self, mile_stones, optimizer, epoch, logger):
        for i in range(0, len(mile_stones)):
            epoch_start = mile_stones[i]['epoch_start']
            epoch_end = mile_stones[i]['epoch_end']
            learning_rate = mile_stones[i]['learning_rate']
            if epoch_start <= epoch < epoch_end:
                lr = learning_rate

        logger.info('Epoch [{}] learning rate = {}'.format(epoch, lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def validate(self, args, test_loader, model, criterion, logger):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        model.eval()
        end = time.time()
        #print('------------validation------------')
        for i, (input, target) in enumerate(test_loader):
            # print('before', target.size(), target.dtype)
            target = target.squeeze().view(-1).to(args.device)
            # print('after', target.size(), target.dtype)
            input = input.to(args.device)

            output, fea = model(input)

            loss = criterion(output, target)
            losses.update(loss.item(), input.size(0))

            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        logger.info(
            "Loss {loss.avg:.3f}\t"
            "Prec@1 {top1.avg:.3f}\t".format(
                loss=losses,
                top1=top1
            )
        )
        logger.info(' * Prec@1 \t{top1.avg:.3f}\t'.format(top1=top1))
        model.train()
        return top1.avg

    def validate_attack(self, args, test_loader, model, criterion, logger):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        model.eval()
        end = time.time()
        for i, (inputs, target, index_class) in enumerate(test_loader):
            # print('before', target.size(), target.dtype)
            if len(inputs.size()) == 1:
                inputs = torch.unsqueeze(inputs, 1)
            target = target.squeeze().view(-1).to(args.device)
            # print('after', target.size(), target.dtype)
            inputs = inputs.to(args.device)

            output = model(inputs)

            loss = criterion(output, target)
            losses.update(loss.item(), inputs.size(0))

            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], inputs.size(0))

            batch_time.update(time.time() - end)
            end = time.time()
        logger.info("Loss {loss.avg:.3f}\t"
                    "Prec@1 {top1.avg:.3f}\t".format(
            loss=losses,
            top1=top1))
        logger.info(' * Prec@1 \t{top1.avg:.3f}\t'.format(top1=top1))
        model.train()
        return top1.avg

    @torch.no_grad()
    def get_predictions(self, args, model, dataloader):
        preds = []
        targets = []
        model = model.to(args.device)
        model.eval()
        for i, (inputs, target) in enumerate(dataloader):
            # inference
            target = target.squeeze().long().to(args.device)
            inputs = inputs.to(args.device)
            output, fea = model(inputs)
            ''' softmax could be important '''
            output = F.softmax(output, dim=-1)
            # add to list
            preds += list(output.cpu())
            targets += list(target.cpu())
            torch.cuda.empty_cache()
        return preds, targets

    @torch.no_grad()
    def get_predictions_test(self, args, model, dataloader):
        preds = []
        targets = []
        #groups = []
        model = model.to(args.device)
        model.eval()
        for i, (inputs, target, group_index) in enumerate(dataloader):
        #for i, (inputs, target) in enumerate(dataloader):
            # inference
            target = target.squeeze().long().to(args.device)
            inputs = inputs.to(args.device)
            output, fea = model(inputs)
            ''' softmax could be important '''
            output = F.softmax(output, dim=-1)
            # add to list
            preds += list(output.cpu())
            targets += list(target.cpu())
            #groups += list(group_index.cpu())
            torch.cuda.empty_cache()
        model = model.cpu()
        torch.cuda.empty_cache()
        return preds, targets#, groups

    def load_epoch_conf(self, args, model_type='target', option='train'):
        conf_file = args.epoch_file
        if model_type == 'target' or model_type == 'shadow':
            if option == 'train':
                conf_path = os.path.join(settings.CONF_TRAIN_DIR, '{}.yml'.format(conf_file))
            else:
                conf_path = os.path.join(settings.CONF_RETRAIN_DIR, '{}.yml'.format(conf_file))
        else:
            conf_path = os.path.join(settings.CONF_TRAIN_ATTACK_DIR, '{}.yml'.format(conf_file))

        # open
        with open(file=conf_path, mode="rb") as f:
            infos = yaml.load(f, Loader=yaml.FullLoader)
        # add to args
        args.lr = infos['mile_stone'][0]['learning_rate']
        args.start_epoch = infos['start_epoch']
        args.epoch = infos['epoch']
        args.mile_stone = infos['mile_stone']
        return args

    def produce_attack(self):
        pass

    def get_target_model(self):
        return self.target_model

    def get_shadow_models(self):
        return self.shadow_models

    def get_shadow_model(self, idx):
        return self.shadow_models[idx]

    def get_attack_model(self):
        return self.attack_model
