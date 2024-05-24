import torch
from models import model_selector
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_target_model(floder_path, dataset, arch, arch_conf, if_pruned=True):
    parser = argparse.ArgumentParser(description='PyTorch load model')
    args = parser.parse_args()
    args.dataset = dataset
    args.arch = arch
    args.arch_conf = arch_conf
    #model = model_selector.get_network(args)
    if if_pruned:
        model = torch.load(f'{floder_path}/target_pruned/model_latest_target_pruned.path.tar')
    else:
        model = torch.load(f'{floder_path}/target/model_latest_target.path.tar')
    #model.load_state_dict(checkpoint['state_dict'])
    #new_model = model_selector.get_network(args)
    #print(list(model.parameters())[0] == list(new_model.parameters())[0])
    #new_model.load_state_dict(checkpoint['state_dict'])
    #print(list(model.parameters())[0])
    #print(list(model.parameters())[0] == list(new_model.parameters())[0])
    return model


def load_attack_model(floder_path, num_classes):
    #model = get_attack_model(num_classes)
    model = torch.load(f'{floder_path}/attack/model_latest_attack.path.tar')
    #a = list(model.parameters())[0]
    #model.load_state_dict(checkpoint['state_dict'])
    #print(a == list(model.parameters())[0])
    return model


def get_attack_model(num_classes, arch='linear'):
    """ return given model
    """
    dim_input = num_classes * 2
    from membership_attack.model.attack_model_linear import AttackNetLinear
    net = AttackNetLinear(class_num=dim_input)
    return net
