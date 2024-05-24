from conf import settings
import os
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset
import json


def dataloader_selector(dataset='cifar100', k=5, split_train=0.):
    dataloaders = {'train': None, 'test': None, 'val': None}
    infos = load_dataset_conf(dataset)
    infos['split_train'] = split_train
    data_dir = os.path.join(settings.DATASET_DIR, infos['relative_path'])
    if dataset == 'cifar10' or dataset == 'cifar10_test':
        print(dataset)
        from dataloader import data_cifar10
        dataloaders['target_train'], dataloaders['target_test'], dataloaders['target_ref'], \
            dataloaders['shadow_train'], dataloaders['shadow_test'], dataloaders['shadow_ref'] \
            = data_cifar10.prepare_cifar10_dataset(
            data_dir, k=k,
            options={'group': infos['group'], 'split_train': infos['split_train']}
        )
    elif dataset == 'cifar100' or dataset == 'cifar100_test':
        print(dataset)
        from dataloader import data_cifar100
        dataloaders['target_train'], dataloaders['target_test'], dataloaders['target_ref'], \
            dataloaders['shadow_train'], dataloaders['shadow_test'], dataloaders['shadow_ref'] \
            = data_cifar100.prepare_cifar100_dataset(
            data_dir, k=k,
            options={'group': infos['group'], 'split_train': infos['split_train']}
        )
    elif dataset in ['svhn', 'svhn_test']:
        print(dataset)
        from dataloader import data_svhn
        dataloaders['target_train'], dataloaders['target_test'], dataloaders['target_ref'], \
            dataloaders['shadow_train'], dataloaders['shadow_test'], dataloaders['shadow_ref'] \
            = data_svhn.prepare_svhn_dataset(
            data_dir, k=k,
            options={'group': infos['group'], 'split_train': infos['split_train']}
        )
    elif dataset in ['arxiv10', 'arxiv10_test']:
        print(dataset)
        from dataloader import data_arxiv10
        dataloaders['target_train'], dataloaders['target_test'], dataloaders['target_ref'], \
        dataloaders['shadow_train'], dataloaders['shadow_test'], dataloaders['shadow_ref'] \
            = data_arxiv10.prepare_arxiv10_dataset(
            data_dir, k=k,
            options={'group': infos['group'], 'split_train': infos['split_train']}
        )
    else:
        raise NotImplementedError
    return dataloaders


def load_dataset_conf(dataset):
    conf_path = os.path.join(settings.CONF_DIR, '{}.yml'.format(dataset))
    with open(file=conf_path, mode="rb") as f:
        infos = yaml.load(f, Loader=yaml.FullLoader)
    return infos


if __name__ == '__main__':
    dataloaders = dataloader_selector()
    print(dataloaders['train'])
