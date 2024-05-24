import logging
import os
import time
import torch
from torch import nn

from mia.vanilla import VanillaMia
from representation.aug.aug_selector import get_aug
from representation.loss.crl import CRL
from conf import settings
from conf.model_conf import dim_fea
from dataloader.info import num_classes

from util.averagemeters import AverageMeter
from util.metrics import accuracy
from util.yaml_loader import load_loss_conf


class VanillaMia_CRL(VanillaMia):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(VanillaMia_CRL, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref, shadow_ref
        )
        self.producer = None
        self.loss_confs = None

    def load_components(self, args):
        from representation.aug.aug_selector import get_aug
        self.producer = get_aug(args)
        self.loss_confs = load_loss_conf(args.loss_conf)

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
        criterion = CRL(num_classes[args.dataset], dim_fea[args.arch], size_average=True, **self.loss_confs).to(args.device)
        #criterion = RelaxCL(num_classes[args.dataset], num_classes[args.dataset], size_average=True, **self.loss_confs).to(args.device)
        ce = nn.CrossEntropyLoss().to(args.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        optimizer_cl = torch.optim.SGD(
            criterion.parameters(),
            lr=self.loss_confs['lr_c'],
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        ''' training '''
        best_prec1 = 0
        cur_prec1 = 0
        step = 0
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            #optimizer_cl = self.adjust_learning_rate_cl(args.mile_stone, optimizer_cl, current_epoch, logger)
            self.classification_train_cl(
                args,
                model=model, dataloader=self.target_train,
                optimizer=optimizer, optimizer_cl=optimizer_cl, criterion=criterion,
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
        ''' others '''
        criterion = CRL(num_classes=num_classes[args.dataset], fea_dim=dim_fea[args.arch], size_average=True, **self.loss_confs).to(args.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        optimizer_cl = torch.optim.SGD(
            criterion.parameters(),
            lr=self.loss_confs['lr_c'],
            momentum=args.momentum,
            weight_decay=args.weight_decay
        )
        ce = nn.CrossEntropyLoss().to(args.device)
        ''' training '''
        best_prec1 = 0
        cur_prec1 = 0
        step = 0
        for current_epoch in range(args.start_epoch, args.epoch):
            optimizer = self.adjust_learning_rate(args.mile_stone, optimizer, current_epoch, logger)
            #optimizer_cl = self.adjust_learning_rate_cl(args.mile_stone, optimizer_cl, current_epoch, logger)
            self.classification_train_cl(
                args,
                model=model, dataloader=train_loader,
                optimizer=optimizer, optimizer_cl=optimizer_cl, criterion=criterion,
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

    def classification_train_cl(self, args, model, dataloader, optimizer, optimizer_cl, criterion, current_epoch, logger):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        total_losses = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        for i, (inputs, target) in enumerate(dataloader):
            data_time.update(time.time() - end)
            target = target.view(-1).long().to(args.device)
            inputs = inputs.to(args.device)
            inputs = self.data_aug(inputs)
            output, fea = model(inputs)
            #print(output.size(), fea.size())
            loss = criterion(output, fea, target, current_epoch+1)
            total_loss = loss
            loss.item()
            losses.update(loss.item(), inputs.size(0))
            total_losses.update(total_loss.item(), inputs.size(0))
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0], inputs.size(0))
            optimizer.zero_grad()
            optimizer_cl.zero_grad()
            total_loss.backward()
            optimizer.step()
            #for param in criterion.parameters():
            #    # lr_cent is learning rate for center loss, e.g. lr_cent = 0.5
            #    param.grad.data *= ( 1.0 / (0.001))
            optimizer_cl.step()
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

    def adjust_learning_rate_cl(self, mile_stones, optimizer, epoch, logger):
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

if __name__ == '__main__':
    a = VanillaMia(None, None, None, None, None, None, None)
