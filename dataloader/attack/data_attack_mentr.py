import torch
from torch.utils.data import Dataset


class MIADataset(Dataset):

    def __init__(
            self,
            data_train_list, data_test_list,
            data_train_class_list, data_test_class_list,
            num_classes=100,
            if_group=False,
            data_train_group_list=None, data_test_group_list=None):
        """

        :param data_train_list: [tensor(num_dim)]
        :param data_test_list: [tensor(num_dim)]
        """
        ''' #classes '''
        self.num_classes = num_classes
        ''' inputs '''
        self.data = []
        self.data += data_train_list
        self.data += data_test_list
        len_train = len(data_train_list)
        # len_test = len(data_test_list)
        len_all = len(self.data)
        ''' classes '''
        self.label = []
        self.label += data_train_class_list
        self.label += data_test_class_list
        ''' member or non-member '''
        self.out = [torch.tensor([1 if i < len_train else 0]) for i in range(len_all)]
        ''' groups '''
        self.group = None
        if if_group:
            self.group = []
            self.group += data_train_group_list
            self.group += data_test_group_list
        print("size of data:", len(self.data), len(self.label), len(self.out))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_p1 = self.data[index]
        index_class = self.label[index]
        data_p2 = self.__num_to_one_hot__(index_class)
        data = data_p1#torch.cat((data_p1, data_p2), dim=0)
        if_member = self.out[index]
        if self.group is not None:
            return data, if_member, self.group[index]
        #print(index)
        return data, if_member, index_class

    def __num_to_one_hot__(self, index_class):
        index_class = index_class.view(-1)
        # print(index_class.size())
        src = torch.ones(1, self.num_classes).long()
        one_hot = torch.zeros(1, self.num_classes).long()
        # print(one_hot.size())
        one_hot.scatter_(
            dim=1, index=index_class.unsqueeze(dim=1), src=src
        )
        one_hot = one_hot.squeeze().float()
        return one_hot


if __name__ == '__main__':
    a = torch.tensor([0 for i in range(10)])
    b = torch.tensor([1 for i in range(10)])
    print(torch.cat((a, b), 0))
    print(list(torch.cat((a, b), 0)))

    a = torch.tensor([[0, 0] for i in range(10)])
    # print(a)
    print(list(a))

    label = [torch.tensor([1 if i < 5 else 0]) for i in range(10)]
    print(label[0].size())
    target = label[0]
    src = torch.ones(1, 5).long()
    one_hot = torch.zeros(1, 5).long()
    one_hot.scatter_(
        dim=1, index=target.unsqueeze(dim=1), src=src
    )
    one_hot = one_hot.squeeze().float()
    print(one_hot)
