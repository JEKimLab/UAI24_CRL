"""
some function are derived from: https://github.com/giangnguyen2412/VanillaGan-DCGAN-for-MNIST-CIFAR10-celebA/blob/master/gan.py
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
from torch.autograd import Variable

from mia.vanilla import VanillaMia
from representation.aug.aug_selector import get_aug
from conf import settings
from dataloader.info import num_classes

from util.averagemeters import AverageMeter
from util.metrics import accuracy
from util.yaml_loader import load_loss_conf

from dataloader.info import image_size


class VanillaMia_DMP_GAN(VanillaMia):
    def __init__(
            self,
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list,
            target_ref=None, shadow_ref=None
    ):
        super(VanillaMia_DMP_GAN, self).__init__(
            target_model, shadow_models, attack_model,
            target_train, target_test, shadow_train_list, shadow_test_list
        )
        # models
        self.target_model_up = copy.deepcopy(self.target_model)
        self.shadow_models_up = []
        for m in self.shadow_models:
            self.shadow_models_up += [copy.deepcopy(m)]
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
        self.num_ref = self.loss_confs['num_ref']

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
        # Initialize generator and discriminator
        model_arch = 'dc_gan'
        model_type = args.dataset
        # GAN & Loss function
        adversarial_loss = torch.nn.BCELoss().to(args.device)
        generator = Generator(model_arch=model_arch, model_type=model_type).to(args.device)
        discriminator = Discriminator(model_arch=model_arch, model_type=model_type).to(args.device)
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # Train GAN
        for current_epoch in range(200):
            self.gan_train(
                args, generator, discriminator, self.target_train,
                adversarial_loss, optimizer_G, optimizer_D,
                current_epoch, logger
            )
        # Save GAN
        final_path = os.path.join(save_path, 'target')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}.path.tar'.format('target_g'))
        # torch.save(generator.cpu(), final_path)
        final_path = os.path.join(save_path, 'target')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}.path.tar'.format('target_d'))
        # torch.save(discriminator.cpu(), final_path)
        # Produce Ref
        generator.eval()
        model = model.to(args.device)
        model.eval()
        # Sample noise as generator input
        num_ref_batch = self.num_ref // args.batch_size
        data_x_temp = []
        data_y_temp = []
        for i in range(num_ref_batch):
            z = torch.randn(args.batch_size, 100, 1, 1).to(args.device)
            # Generate a batch of images
            gen_imgs = generator(z)
            out, fea = model(gen_imgs)
            data_x_temp += [gen_imgs.data.cpu().numpy()]
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
        #ref_loader = self.shadow_ref_list[idx]
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
        # Initialize generator and discriminator
        model_arch = 'dc_gan'
        model_type = args.dataset
        # GAN & Loss function
        adversarial_loss = torch.nn.BCELoss().to(args.device)
        generator = Generator(model_arch=model_arch, model_type=model_type).to(args.device)
        discriminator = Discriminator(model_arch=model_arch, model_type=model_type).to(args.device)
        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        # Train GAN
        for current_epoch in range(200):
            self.gan_train(
                args, generator, discriminator, train_loader,
                adversarial_loss, optimizer_G, optimizer_D,
                current_epoch, logger
            )
        # Save GAN
        final_path = os.path.join(save_path, 'shadow')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}_{}.path.tar'.format('shadow_g', idx))
        # torch.save(generator.cpu(), final_path)
        final_path = os.path.join(save_path, 'shadow')
        if not os.path.exists(final_path):
            os.makedirs(final_path)
        final_path = os.path.join(final_path, 'model_latest_{}_{}.path.tar'.format('shadow_d', idx))
        # torch.save(discriminator.cpu(), final_path)
        # Produce Ref
        generator.eval()
        model = model.to(args.device)
        model.eval()
        # Sample noise as generator input
        num_ref_batch = self.num_ref // args.batch_size
        data_x_temp = []
        data_y_temp = []
        for i in range(num_ref_batch):
            z = torch.randn(args.batch_size, 100, 1, 1).to(args.device)
            # Generate a batch of images
            gen_imgs = generator(z)
            out, fea = model(gen_imgs)
            data_x_temp += [gen_imgs.data.cpu().numpy()]
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
        final_path = os.path.join(final_path, 'model_latest_{}_{}.path.tar'.format('shadow', idx))
        torch.save(model.cpu(), final_path)

    def gan_train(self, args, generator, discriminator, dataloader, adversarial_loss, optimizer_G, optimizer_D,
                  current_epoch, logger):
        g_losses = AverageMeter()
        d_losses = AverageMeter()
        Tensor = torch.cuda.FloatTensor
        generator.train()
        discriminator.train()
        for i, (imgs, _) in enumerate(dataloader):
            # Adversarial ground truths
            valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
            # -----------------
            #  Train Generator
            # -----------------
            B = imgs.size(0)
            for j in range(3):
                optimizer_G.zero_grad()
                # Sample noise as generator input
                z = torch.randn(imgs.shape[0], 100, 1, 1).to(args.device)
                # Generate a batch of images
                gen_imgs = generator(z)
                #print(gen_imgs.size())
                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs).view(B, -1), valid)
                g_loss.backward()
                optimizer_G.step()
                g_losses.update(g_loss.item(), imgs.size(0))
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            # Measure discriminator's ability to classify real from generated samples
            cache = discriminator(real_imgs)
            real_loss = adversarial_loss(cache.view(B, -1), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()).view(B, -1), fake)
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_D.step()
            d_losses.update(d_loss.item(), imgs.size(0))

            if i % args.print_freq == 0:
                logger.info("GAN Epoch: [{0}]\t"
                            "Iter: [{1}]\t"
                            "GLoss {gloss.val:.3f} ({gloss.avg:.3f})\t"
                            "DLoss {dloss.val:.3f} ({dloss.avg:.3f})\t".format(
                    # "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                    current_epoch,
                    i,
                    gloss=g_losses,
                    dloss=d_losses,
                    # top1=top1
                ))

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
            # print(output.size(), target.size())
            loss = F.kl_div(F.log_softmax(output / tau, dim=-1), F.softmax(target / tau, dim=-1), reduction='mean')
            total_loss = loss
            loss.item()
            losses.update(loss.item(), inputs.size(0))
            total_losses.update(total_loss.item(), inputs.size(0))
            # prec1 = accuracy(output.data, target, topk=(1,))
            # top1.update(prec1[0], inputs.size(0))
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
                    # "Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t".format(
                    current_epoch,
                    i,
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=total_losses,
                    # top1=top1
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


class Generator(nn.Module):
    def __init__(self, model_arch='vanilla_gan', model_type='cifar10'):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True, bias=True):
            layers = [nn.Linear(in_feat, out_feat, bias=bias)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model_type = model_type
        self.model_arch = model_arch
        self.image_shape = {'mnist': (1, 32, 32),
                            'cifar10': (3, 32, 32),
                            'celebA': (3, 64, 64)}

        if self.model_type == 'mnist':
            channels = 1
            img_size = 32
        elif self.model_type == 'cifar10' or self.model_type == 'cifar100':
            channels = 3
            img_size = 32
        elif self.model_type == 'celebA':
            channels = 3
            img_size = 64
        else:
            print("Error: Not defined")
            return

        self.init_size = img_size // 4

        if self.model_arch == 'dc_gan':
            self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.models_cifar10 = nn.ModuleDict({

            'vanilla_gan': nn.Sequential(
                *block(100, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                *block(1024, 2048),
                *block(2048, 2048),
                *block(2048, 1024),
                nn.Linear(1024, 3 * 64 * 64),
                nn.Tanh()),

            'dc_gan': nn.Sequential(
                nn.ConvTranspose2d(100, 256, 4, 1, 0, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(True),
                nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(True),
                nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                #nn.BatchNorm2d(64),
                #nn.ReLU(True),
                #nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                nn.Tanh()),
        })

        self.models_mnist = nn.ModuleDict({

            'vanilla_gan': nn.Sequential(
                *block(100, 128, normalize=False),
                *block(128, 256),
                *block(256, 512),
                *block(512, 1024),
                nn.Linear(1024, channels * img_size * img_size),
                nn.Tanh()),

            'dc_gan': nn.Sequential(
                nn.Conv2d(100, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(128, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.BatchNorm2d(64, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(64, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Upsample(scale_factor=2),
                nn.Conv2d(32, 16, 3, stride=1, padding=1),
                nn.BatchNorm2d(16, 0.8),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(16, channels, 3, stride=1, padding=1),
                nn.Tanh())

        })

        self.models_celebA = nn.ModuleDict({

            'dc_gan': nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        })

        self.model = {'mnist': self.models_mnist,
                      'cifar10': self.models_cifar10,
                      'cifar100': self.models_cifar10,
                      'celebA': self.models_celebA}

    def forward(self, z):
        if self.model_arch == 'vanilla_gan':
            z = z.view(z.size(0), -1)
        img = self.model[self.model_type][self.model_arch](z)
        if self.model_arch == 'vanilla_gan':
            img = img.view(img.size(0), *self.image_shape[self.model_type])
        return img


class Discriminator(nn.Module):
    def __init__(self, model_arch='dc_gan', model_type='cifar10'):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model_type = model_type
        self.model_arch = model_arch
        self.image_shape = {'mnist': (1, 32, 32),
                            'cifar10': (3, 32, 32),
                            'celebA': (3, 64, 64)}

        if self.model_type == 'mnist':
            channels = 1
            img_size = 32
        elif self.model_type == 'cifar10' or self.model_type == 'cifar100':
            channels = 3
            img_size = 32
        elif self.model_type == 'celebA':
            channels = 3
            img_size = 64
        else:
            print("Channel size is not defined")
            return

        self.init_size = img_size // 4

        if self.model_arch == 'dc_gan':
            ds_size = img_size // 2 ** 4
            self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

        self.models_cifar10 = nn.ModuleDict({

            'vanilla_gan': nn.Sequential(
                nn.Linear(3 * 32 * 32, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()),

            'dc_gan': nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(256, 1, 4, 1, 0, bias=False),
                #nn.BatchNorm2d(512),
                #nn.LeakyReLU(0.2, inplace=True),
                #nn.Conv2d(512, 1, 4, 1, 0, bias=False),
                nn.Sigmoid())
        })

        self.models_mnist = nn.ModuleDict({

            'vanilla_gan': nn.Sequential(
                nn.Linear(channels * img_size * img_size, 512),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 256),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 1),
                nn.Sigmoid()),

            'dc_gan': nn.Sequential(
                *discriminator_block(channels, 16, bn=False),
                *discriminator_block(16, 32),
                *discriminator_block(32, 64),
                *discriminator_block(64, 128))
        })

        self.models_celebA = nn.ModuleDict({

            'dc_gan': nn.Sequential(

                # input is (nc) x 64 x 64
                nn.Conv2d(3, 64, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 2),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 4),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(64 * 8),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()

            )
        })

        self.model = {'mnist': self.models_mnist,
                      'cifar10': self.models_cifar10,
                      'cifar100': self.models_cifar10,
                      'celebA': self.models_celebA}

    def forward(self, img):
        if self.model_arch == 'dc_gan':
            out = self.model[self.model_type][self.model_arch](img)
            if self.model_type == 'mnist':
                out = out.view(out.shape[0], -1)
                return self.adv_layer(out)
            return out

        elif self.model_arch == 'vanilla_gan':
            out = img.view(img.shape[0], -1)
            validity = self.model[self.model_type][self.model_arch](out)

        return validity


if __name__ == '__main__':
    a = VanillaMia(None, None, None, None, None, None, None)
