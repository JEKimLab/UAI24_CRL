import pandas as pd
import torch.utils.data
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np

classes = {
    'astro-ph': 0, 'cond-mat': 1, 'cs': 2, 'eess': 3, 'hep-ph': 4,
    'hep-th': 5, 'math': 6, 'physics': 7, 'quant-ph': 8, 'stat': 9
}


class ArXiv10Dataset(Dataset):
    def __init__(self, X, Y, group=None):
        # read in the transforms
        self.if_show_group = True if group is not None else False
        #
        self.data = X
        self.labels = Y
        self.groups = Y

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
        if self.if_show_group:
            groups = self.groups[index]
            return data, labels, groups
        return data, labels


def prepare_arxiv10_dataset(data_dir, k, options=None):
    if options is None:
        options = {'group': None, 'split_train': 0.0}
    split_train = options['split_train']

    data = np.load(f'{data_dir}/data.npz')
    X, Y = data['data'], data['label']

    # Target Dataset
    target_x, shadow_x, target_y, shadow_y = train_test_split(
        X, Y, test_size=0.5, random_state=0
    )
    target_train_x, target_test_x, target_train_y, target_test_y = train_test_split(
        target_x, target_y, test_size=0.5, random_state=0
    )
    # Target Ref
    target_ref = None
    if split_train > 0:
        target_train_x, target_ref_x, target_train_y, target_ref_y = train_test_split(
            target_train_x, target_train_y, test_size=split_train, random_state=0
        )
        target_ref = ArXiv10Dataset(
            target_ref_x, target_ref_y,
            group=options['group']
        )
    target_train = ArXiv10Dataset(
        target_train_x, target_train_y,
        group=options['group']
    )
    target_test = ArXiv10Dataset(
        target_test_x, target_test_y,
        group=options['group']
    )
    # Shadow Dataset
    shadow_train_list = []
    shadow_test_list = []
    shadow_ref_list = []
    for i in range(k):
        shadow_train_x, shadow_test_x, shadow_train_y, shadow_test_y = train_test_split(
            shadow_x, shadow_y, test_size=0.5, random_state=i
        )
        if split_train > 0:
            shadow_train_x, shadow_ref_x, shadow_train_y, shadow_ref_y = train_test_split(
                shadow_train_x, shadow_train_y, test_size=split_train, random_state=i
            )
            shadow_ref_list += [
                ArXiv10Dataset(
                    shadow_ref_x, shadow_ref_y,
                    group=options['group']
                )
            ]
        shadow_train_list += [
            ArXiv10Dataset(
                shadow_train_x, shadow_train_y,
                group=options['group']
            )
        ]
        shadow_test_list += [
            ArXiv10Dataset(
                shadow_test_x, shadow_test_y,
                group=options['group']
            )
        ]
    return target_train, target_test, target_ref, shadow_train_list, shadow_test_list, shadow_ref_list


if __name__ == '__main__':
    target_train, target_test, target_ref, shadow_train_list, shadow_test_list, shadow_ref_list = prepare_arxiv10_dataset(
        data_dir='/home/xfang23/dataset/ArXiv-10', k=1, options=None
    )
    print(
        len(target_train), len(target_test), len(shadow_train_list[0]), len(shadow_test_list[0])
    )
    train_loader = DataLoader(dataset=target_train, batch_size=64, shuffle=False)
    for i, (inputs, target) in enumerate(train_loader):
        # print(target)
        if i == 0:
            print(inputs.size())
            print(target.size())
