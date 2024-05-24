import torch


# one-hot --> labels
def onehot_to_index(output):
    labels_again = torch.argmax(output, dim=1)
    return labels_again


# labels --> one-hot
def index_to_onehot(labels):
    one_hot = torch.nn.functional.one_hot(labels)
    return one_hot
