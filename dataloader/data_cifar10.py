import torch
from torch import nn
from torch.utils.data import Dataset, TensorDataset, DataLoader
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from sklearn.model_selection import train_test_split

classes = (
    'plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
)

files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch']

crop_size = 32
padding = 4


class CIFAR10Dataset(Dataset):
    def __init__(self, X, Y, group=None):
        # read in the transforms
        self.if_show_group = True if group is not None else False
        # reshape into 48x48x1
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


def prepare_cifar10_dataset(data_dir, k, options=None):
    if options is None:
        options = {'group': None, 'split_train':0.0}
    split_train = options['split_train']
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        ),
    ])

    # CIFAR10 Train
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024,
        shuffle=False, num_workers=0,
        pin_memory=True
    )
    # CIFAR10 Test
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=1024,
        shuffle=False, num_workers=0,
        pin_memory=True
    )
    x = []
    y = []
    for i, (inputs, target) in enumerate(train_loader):
        x += [inputs.numpy()]
        y += [target.numpy()]
    for i, (inputs, target) in enumerate(test_loader):
        x += [inputs.numpy()]
        y += [target.numpy()]
    X = x[0]
    Y = y[0]
    for i in range(1, len(x)):
        X = np.concatenate((X, x[i]), axis=0)
        Y = np.concatenate((Y, y[i]), axis=0)

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
        target_ref = CIFAR10Dataset(
            target_ref_x, target_ref_y,
            group=options['group']
        )
    # target dataset class
    target_train = CIFAR10Dataset(
        target_train_x, target_train_y,
        group=options['group']
    )
    target_test = CIFAR10Dataset(
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
                CIFAR10Dataset(
                    shadow_ref_x, shadow_ref_y,
                    group=options['group']
                )
            ]
        shadow_train_list += [
            CIFAR10Dataset(
                shadow_train_x, shadow_train_y,
                group=options['group']
            )
        ]
        shadow_test_list += [
            CIFAR10Dataset(
                shadow_test_x, shadow_test_y,
                group=options['group']
            )
        ]

    return target_train, target_test, target_ref, shadow_train_list, shadow_test_list, shadow_ref_list




def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict



if __name__ == '__main__':
    batch = unpickle('/home/xfang23/dataset/cifar-10-batches-py/data_batch_1')
    #print(batch)
    print(batch[b'labels'][0])
    print(len(batch[b'data'][0]))

    options = {
        'group': 'ethnicity'
    }
    target_train, target_test, shadow_train_list, shaow_test_list = load_cifar10('/home/xfang23/dataset',
                                                                                        k=1, options=options)
    train_loader = DataLoader(dataset=target_train, batch_size=64, shuffle=False)
    for i, (inputs, target, group) in enumerate(train_loader):
        # print(target)
        if i == 0:
            print(inputs)
            print(target)
            print(group)
