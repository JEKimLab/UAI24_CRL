import argparse
import os
import yaml

from functional_modules import train_mia
from conf import settings


def parse_args():
    # hyper-parameters are from ResNet paper
    """
    you can find where to set the following hyper-parameters in folder ./conf:
    [1] ./conf/train
        1   epochs & start epoch
        2   learning rates
        3   momentum & weight decay
    [2] ./conf/dataset
        define how to load your dataset here
    [3] ./conf/model
        define your model configuration here
    """
    parser = argparse.ArgumentParser(description='PyTorch training')
    parser.add_argument('cmd', choices=['train', 'test'])
    parser.add_argument('arch', metavar='ARCH', default='mobilenet',
                        help='model architecture')
    parser.add_argument('--arch-conf', metavar='ARCH-CONF', default='mobilenet_50',
                        help='model architecture')
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'object_detection'])
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--print-freq', default=300, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to  latest checkpoint (default: None)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pretrained model')
    parser.add_argument('--step-ratio', default=0.1, type=float,
                        help='ratio for learning rate deduction')
    parser.add_argument('--warm-up', action='store_true',
                        help='for n = 18, the model needs to warm up for 400 '
                             'iterations')
    parser.add_argument('--save-folder', default='save_checkpoints/', type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--load-folder', default='save_checkpoints/', type=str,
                        help='folder to load the target and shadows')
    parser.add_argument('--summary-folder', default='runs/', type=str,
                        help='folder to save the summary')
    parser.add_argument('--eval-every', default=1, type=int,
                        help='evaluate model every (default: 1) iterations')

    # dataset settings
    parser.add_argument('--dataset', '-d', type=str, default='cifar100',
                        help='dataloader choice')
    parser.add_argument('--type-options', type=str, default='age',
                        help='muti-type classes dataset such as utk')
    parser.add_argument('--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4 )')

    # train conf
    parser.add_argument('--batch-size', default=128, type=int,
                        help='mini-batch size (default: 128)')
    parser.add_argument('--attack-batch-size', default=256, type=int,
                        help='mini-batch size (default: 256)')
    parser.add_argument('--epoch-file', default='cifar100_mobilenet', type=str,
                        help='training file')
    parser.add_argument('--loss-conf', type=str, default='')
    parser.add_argument('--if-restore', type=str, default='no')
    parser.add_argument(
        '--train-target', type=str, default='yes',
        help='if train target model (default: yes)'
    )
    parser.add_argument(
        '--train-shadow', type=str, default='yes',
        help='if train shadow models (default: yes)'
    )
    # pruning things
    parser.add_argument('--if-pruning', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--pruning-rate', default=0.9, type=float, help='pruning rate, range [0, 1)')
    parser.add_argument('--pruning-method', type=str, default='l1_filter')
    # parser.add_argument('--if-retrain', type=str, default='yes', choices=['yes', 'no'])

    # train_attack things
    # parser.add_argument('--if-attack', type=str, default='no', choices=['yes', 'no'])
    parser.add_argument('--learn-method', type=str, default='vanilla')
    parser.add_argument('--attack-method', type=str, default='vanilla')
    parser.add_argument('--k', default=1, type=int,
                        help='num of shadow models')
    parser.add_argument('--attack-arch', type=str, default='linear')


    args = parser.parse_args()
    return args


def load_epoch_conf(conf_file):
    conf_path = os.path.join(settings.CONF_TRAIN_DIR, '{}.yml'.format(conf_file))
    with open(file=conf_path, mode="rb") as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)
    return infos


def main():
    args = parse_args()
    """
    load epochs and optimizer settings from ./conf/train 
    """
    # training settings
    info = load_epoch_conf(args.epoch_file)
    # conf: epochs
    args.lr = info['mile_stone'][0]['learning_rate']
    args.start_epoch = info['start_epoch']
    args.epoch = info['epoch']
    args.mile_stone = info['mile_stone']
    # conf: optimizer
    args.momentum = info['momentum']
    args.weight_decay = info['weight_decay']
    """
    go to the specific task
    """
    train_mia.main(args)


if __name__ == '__main__':
    main()
