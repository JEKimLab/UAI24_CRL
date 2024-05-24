"""
some function are derived from: https://github.com/DingfanChen/RelaxLoss/blob/main/source/cifar/defense/advreg.py
"""
import logging
import os
import time

import numpy as np
import torch
from torch import nn

from mia.vanilla import VanillaMia
from representation.aug.aug_selector import get_aug
from representation.loss.relaxloss import RelaxLoss
from conf import settings
from dataloader.info import num_classes

from util.averagemeters import AverageMeter
from util.metrics import accuracy
from util.yaml_loader import load_loss_conf


class VanillaMia_AdvReg(VanillaMia):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
    ):
        super(VanillaMia_AdvReg, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
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
        self.alpha = self.loss_confs['alpha']

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
        model = self.target_model.to(args.device)
        ''' others '''
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        model_ref = Attack(input_dim=self.num_class).to(args.device)
        optimizer_ref = torch.optim.Adam(
            model_ref.parameters(),
            lr=0.001,
        )
        criterion = nn.CrossEntropyLoss().to(args.device)
        criterion_ref = nn.BCELoss().to(args.device)
        ce = nn.CrossEntropyLoss().to(args.device)
        ''' training '''
        best_prec1 = 0
        cur_prec1 = 0
        step = 0
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            self.classification_train_advreg(
                args,
                model=model, model_ref=model_ref,
                dataloader=self.target_train, dataloader_ref=self.target_ref,
                optimizer=optimizer, optimizer_ref=optimizer_ref,
                criterion=criterion, criterion_ref=criterion_ref,
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
        model = self.shadow_models[idx].to(args.device)
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
        model_ref = Attack(input_dim=self.num_class).to(args.device)
        optimizer_ref = torch.optim.Adam(
            model_ref.parameters(),
            lr=0.001,
        )
        criterion = nn.CrossEntropyLoss().to(args.device)
        criterion_ref = nn.BCELoss().to(args.device)
        ce = nn.CrossEntropyLoss().to(args.device)
        ''' training '''
        best_prec1 = 0
        cur_prec1 = 0
        step = 0
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            self.classification_train_advreg(
                args,
                model=model, model_ref=model_ref,
                dataloader=train_loader, dataloader_ref=ref_loader,
                optimizer=optimizer, optimizer_ref=optimizer_ref,
                criterion=criterion, criterion_ref=criterion_ref,
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
        final_path = os.path.join(final_path, 'model_latest_{}_{}.path.tar'.format('shadow', idx))
        torch.save(model.cpu(), final_path)

    def classification_train_advreg(self, args, model, model_ref, dataloader, dataloader_ref, optimizer, optimizer_ref,
                                    criterion, criterion_ref, current_epoch, logger):
        self.train_clf_model(
            args, model, model_ref,
            dataloader, optimizer, criterion,
            current_epoch, logger
        )
        self.train_infer_model(
            args, model, model_ref,
            dataloader, dataloader_ref,
            optimizer_ref, criterion_ref,
            current_epoch, logger
        )

    def train_clf_model(self, args, model, model_ref, dataloader, optimizer, criterion, current_epoch, logger):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        total_losses = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        model.train()
        model_ref.eval()

        for i, (inputs, target) in enumerate(dataloader):
            data_time.update(time.time() - end)
            target = target.view(-1).long().to(args.device)
            inputs = inputs.to(args.device)
            inputs = self.data_aug(inputs)
            # inference
            output, fea = model(inputs)
            inference_input_x, inference_input_y = self.attack_input_transform(output, target)
            #inference_input_x, inference_input_y = inference_input_x.to(args.device), inference_input_y.to(args.device)
            inference_output = model_ref(inference_input_x, inference_input_y)
            # loss
            #print(output.size(), target.size(), inference_output.size())
            loss = criterion(output, target) + (self.alpha * (torch.mean(inference_output) - 0.5))
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

    def train_infer_model(
            self, args, model, model_ref, dataloader, dataloader_ref, optimizer, criterion, current_epoch, logger
    ):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        total_losses = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        model_ref.train()
        model.eval()

        for i, (member, nonmember) in enumerate(zip(dataloader, dataloader_ref)):
            inputs_member, targets_member = member
            inputs_nonmember, targets_nonmember = nonmember
            inputs_member, targets_member = inputs_member.to(args.device), targets_member.to(args.device)
            inputs_nonmember, targets_nonmember = inputs_nonmember.to(args.device), targets_nonmember.to(args.device)
            inputs_member, inputs_nonmember = self.data_aug(inputs_member), self.data_aug(inputs_nonmember)
            out_member, _ = model(inputs_member)
            out_nonmember, _ = model(inputs_nonmember)
            outputs_member_x, outputs_member_y = self.attack_input_transform(
                out_member, targets_member
            )
            outputs_nonmember_x, outputs_nonmember_y = self.attack_input_transform(
                out_nonmember, targets_nonmember
            )
            attack_input_x = torch.cat((outputs_member_x, outputs_nonmember_x))
            attack_input_y = torch.cat((outputs_member_y, outputs_nonmember_y))
            attack_labels = np.zeros((inputs_member.size()[0] + inputs_nonmember.size()[0]))
            attack_labels[:inputs_member.size()[0]] = 1.  # member=1
            attack_labels[inputs_member.size()[0]:] = 0.  # nonmember=0
            #
            indices = np.arange(len(attack_input_x))
            np.random.shuffle(indices)
            attack_input_x = attack_input_x[indices]
            attack_input_y = attack_input_y[indices]
            attack_labels = attack_labels[indices]
            is_member_labels = torch.from_numpy(attack_labels).type(torch.FloatTensor).to(args.device)
            attack_output = model_ref(attack_input_x.detach(), attack_input_y.detach()).view(-1)
            # Record accuracy and loss
            loss_attack = criterion(attack_output, is_member_labels)
            prec1 = accuracy_binary(attack_output.data, is_member_labels.data)
            losses.update(loss_attack.item(), len(attack_output))
            top1.update(prec1.item(), len(attack_output))
            total_losses.update(loss_attack.item(), len(attack_output))
            # backward
            optimizer.zero_grad()
            loss_attack.backward()
            optimizer.step()
            # log
            data_time.update(time.time() - end)
            if i % args.print_freq == 0:
                logger.info("Epoch: [{0}]\t"
                            "Iter: [{1}]\t"
                            "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                            "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                            "Loss {loss.val:.3f} ({loss.avg:.3f})\t"
                            "Attack Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                    current_epoch,
                    i,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=total_losses,
                    top1=top1)
                )

    def attack_input_transform(self, x, y):
        """Transform the input to attack model"""
        out_x = x
        out_x, _ = torch.sort(out_x, dim=1)
        one_hot = torch.from_numpy((np.zeros((y.size(0), self.num_class)) - 1)).cuda().type(
            torch.cuda.FloatTensor)
        out_y = one_hot.scatter_(1, y.type(torch.cuda.LongTensor).view([-1, 1]).data, 1)
        return out_x, out_y


class Attack(nn.Module):
    def __init__(self, input_dim, num_classes=1, hiddens=None, y_hiddens=None):
        super(Attack, self).__init__()
        if hiddens is None:
            hiddens = [100, 1024, 512, 64]
        if y_hiddens is None:
            y_hiddens = [100, 512, 64]
        # x hiddens
        self.layers = []
        for i in range(len(hiddens)):
            if i == 0:
                layer = nn.Linear(input_dim, hiddens[i])
            else:
                layer = nn.Linear(hiddens[i - 1], hiddens[i])
            self.layers.append(layer)
        self.layers = nn.ModuleList(self.layers)
        # y hiddens
        self.y_layers = []
        for i in range(len(y_hiddens)):
            if i == 0:
                layer = nn.Linear(input_dim, y_hiddens[i])
            else:
                layer = nn.Linear(y_hiddens[i - 1], y_hiddens[i])
            self.y_layers.append(layer)
        self.y_layers = nn.ModuleList(self.y_layers)
        self.last_layer = nn.Linear(hiddens[-1]+y_hiddens[-1], num_classes)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        output = x
        output_y = y
        for layer in self.layers:
            output = self.relu(layer(output))
        for layer in self.y_layers:
            output_y = self.relu(layer(output_y))
        output = self.last_layer(torch.cat((output, output_y), 1))
        output = self.sigmoid(output)
        return output


def accuracy_binary(output, target):
    """Computes the accuracy for binary classification"""
    batch_size = target.size(0)

    pred = output.view(-1) >= 0.5
    truth = target.view(-1) >= 0.5
    acc = pred.eq(truth).float().sum(0).mul_(100.0 / batch_size)
    return acc


if __name__ == '__main__':
    a = VanillaMia(None, None, None, None, None, None, None)
