import sys

import torch
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from conf.settings import DATASET_DIR

import os
import time
import logging

from dataloader import data_selector
from models.model_selector import get_network
# from dataloader.classification.data_cifar100 import *
from util.averagemeters import AverageMeter
from util.metrics import accuracy

from tensorboardX import SummaryWriter

from util.yaml_loader import load_loss_conf


def main(args):
    save_path = args.save_path = os.path.join(args.save_folder, args.arch, args.arch_conf)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))

    handlers = [logging.FileHandler(args.logger_file, mode='w'),
                logging.StreamHandler()]
    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers)

    if args.cmd == 'train':
        logging.info('start training {}'.format(args.arch))
        run_training(args)
    else:
        NotImplementedError()
        sys.exit()


def run_training(args):
    cudnn.benchmark = True
    ''' dataset '''
    batch_size = args.batch_size
    num_workers = args.workers
    loss_confs = load_loss_conf(args.loss_conf)
    if args.learn_method == 'advreg' or args.learn_method == 'dmp':
        the_dataset = data_selector.dataloader_selector(
            dataset=args.dataset, k=args.k, split_train=loss_confs['split_train']
        )
    else:
        the_dataset = data_selector.dataloader_selector(
            dataset=args.dataset, k=args.k, split_train=0.0
        )
    # Target
    target_train_set = the_dataset['target_train']
    target_test_set = the_dataset['target_test']
    target_ref_set = None
    if args.learn_method == 'advreg' or args.learn_method == 'dmp':
        target_ref_set = the_dataset['target_ref']
    # shadow
    shadow_train_set_list = the_dataset['shadow_train']
    shadow_test_set_list = the_dataset['shadow_test']
    shadow_ref_set_list = None
    if args.learn_method == 'advreg' or args.learn_method == 'dmp':
        shadow_ref_set_list = the_dataset['shadow_ref']
    ''' target dataset '''
    target_train_loader = torch.utils.data.DataLoader(
        target_train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    target_test_loader = torch.utils.data.DataLoader(
        target_test_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False
    )
    target_ref_loader = None
    if args.learn_method == 'advreg' or args.learn_method == 'dmp':
        target_ref_loader = torch.utils.data.DataLoader(
            target_ref_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )
    ''' shadow models '''
    k = args.k
    ''' shadow datasets '''
    shadow_train_loader_list = []
    shadow_test_loader_list = []
    shadow_ref_loader_list = []
    for i in range(k):
        shadow_train_set = shadow_train_set_list[i]
        shadow_test_set = shadow_test_set_list[i]
        if args.learn_method == 'advreg' or args.learn_method == 'dmp':
            shadow_ref_set = shadow_ref_set_list[i]

        shadow_train_loader_list += [torch.utils.data.DataLoader(
            shadow_train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
        )]
        shadow_test_loader_list += [torch.utils.data.DataLoader(
            shadow_test_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False
        )]
        if args.learn_method == 'advreg' or args.learn_method == 'dmp':
            shadow_ref_loader_list += [torch.utils.data.DataLoader(
                shadow_ref_set,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=False
            )]

    ''' Load Model '''
    save_path = os.path.join(args.load_folder, args.arch, args.arch_conf)
    if args.if_restore == 'yes':
        ''' target model '''
        save_target_path = os.path.join(save_path, 'target')
        target_model = torch.load(f'{save_target_path}/model_latest_target.path.tar')
        ''' shadow models '''
        save_shadow_path = os.path.join(save_path, 'shadow')
        shadow_model_list = [torch.load(f'{save_shadow_path}/model_latest_shadow_{i}.path.tar') for i in range(k)]
    else:
        ''' target model '''
        target_model = get_network(args)
        ''' shadow models '''
        shadow_model_list = [get_network(args) for _ in range(k)]

    ''' get attack models '''
    print("Create Attack Model")
    from mia.models import attack_model_selector
    attack_model = attack_model_selector.get_attack_model(args)
    print("Created Attack Model")
    ''' get MIA manager '''
    from mia.mia_selector import get_mia
    mia_manager = get_mia(
        args,
        target_model=target_model,
        shadow_models=shadow_model_list,
        attack_model=attack_model,
        target_train=target_train_loader,
        target_test=target_test_loader,
        shadow_train_list=shadow_train_loader_list,
        shadow_test_list=shadow_test_loader_list,
        target_ref=target_ref_loader,
        shadow_ref_list=shadow_ref_loader_list
    )
    print("Get MIAs")
    if args.learn_method != 'vanilla':
        mia_manager.load_components(args)
    ''' train target and shadow models '''
    if args.if_restore != 'yes':
        if args.train_target == 'yes':
            mia_manager.train_target_model(args)
        elif args.train_target == 'restore':
            save_target_path = os.path.join(save_path, 'target')
            mia_manager.target_model = torch.load(f'{save_target_path}/model_latest_target.path.tar')
        if args.train_shadow == 'yes' and args.k > 0:
            mia_manager.train_shadow_models(args)
    ''' train attack model '''
    # mia_manager.produce_attack(args)  # make attack train and test set
    if args.k > 0:
        mia_manager.train_attack_model(args)
