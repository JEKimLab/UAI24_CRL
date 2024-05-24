import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.stats import norm, kurtosis, skew

crossentropy = nn.CrossEntropyLoss()


def compute_norm_metrics(gradient, metric='l2'):
    """Compute the metrics"""
    if metric == 'l1':
        return np.linalg.norm(gradient, ord=1)
    elif metric == 'l2':
        return np.linalg.norm(gradient)
    elif metric == 'min':
        return np.linalg.norm(gradient, ord=-np.inf)  ## min(abs(x))
    elif metric == 'max':
        return np.linalg.norm(gradient, ord=np.inf)  ## max(abs(x))
    elif metric == 'mean':
        return np.average(gradient)
    elif metric == 'Skewness':
        return skew(gradient)
    elif metric == 'Kurtosis':
        return kurtosis(gradient)
    # return [l1, l2, Min, Max, Mean, Skewness, Kurtosis]


def gradient_based_attack_wrt_x(args, dataloader, model, name='l2'):
    """Gradient w.r.t. input"""
    model.eval()

    ## store results
    # names = ['l1', 'l2', 'Min', 'Max', 'Mean', 'Skewness', 'Kurtosis']
    target_list = []
    stats = []

    ## iterate over batches
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        out_list = []
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        ## iterate over samples within a batch
        for input, target in zip(inputs, targets):
            input = torch.unsqueeze(input, 0)
            input.requires_grad = True
            output, fea = model(input)
            loss = crossentropy(output, torch.unsqueeze(target, 0))
            model.zero_grad()
            loss.backward()

            ## get gradients
            gradient = input.grad.view(-1).cpu().numpy()
            #print(target_list)
            target_list += [target.detach().cpu().numpy()]
            g = compute_norm_metrics(gradient, metric=name)
            out_list += [g]
        # get statistics
        stats += out_list
        #if stats is None:
        #    stats += compute_norm_metrics(out_list, metric=name)
        #else:
        #    stats = np.concatenate((stats, np.array(compute_norm_metrics(out_list, metric=name))), axis=0)
    stats = np.array(stats)
    target_list = np.array(target_list)
    print(stats.shape, target_list.shape)
    return stats, target_list


def gradient_based_attack_wrt_w(args, dataloader, model, name='l2'):
    """Gradient w.r.t. weights"""
    """Gradient w.r.t. input"""
    model.eval()

    ## store results
    # names = ['l1', 'l2', 'Min', 'Max', 'Mean', 'Skewness', 'Kurtosis']
    target_list = []
    stats = []

    ## iterate over batches
    for batch_idx, (inputs, targets) in enumerate(tqdm(dataloader)):
        out_list = []
        inputs, targets = inputs.to(args.device), targets.to(args.device)

        ## iterate over samples within a batch
        for input, target in zip(inputs, targets):
            input = torch.unsqueeze(input, 0)
            input.requires_grad = True
            output, fea = model(input)
            loss = crossentropy(output, torch.unsqueeze(target, 0))
            model.zero_grad()
            loss.backward()

            ## get gradients
            grads_onesample = []
            for param in model.parameters():
                grads_onesample.append(param.grad.view(-1))
            gradient = torch.cat(grads_onesample)
            gradient = gradient.cpu().numpy()
            # print(target_list)
            target_list += [target.detach().cpu().numpy()]
            g = compute_norm_metrics(gradient, metric=name)
            out_list += [g]
        # get statistics
        stats += out_list
        # if stats is None:
        #    stats += compute_norm_metrics(out_list, metric=name)
        # else:
        #    stats = np.concatenate((stats, np.array(compute_norm_metrics(out_list, metric=name))), axis=0)
    stats = np.array(stats)
    target_list = np.array(target_list)
    print(stats.shape, target_list.shape)
    return stats, target_list
