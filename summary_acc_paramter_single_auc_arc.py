import argparse

import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
from summarization import load_model
from summarization.metric_generator import MetricGenerator

batch_size = 256
num_workers = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def get_attack_preds(model, dataloader):
    prob_list = None
    pred_list = None
    true_list = None
    for i, (inputs, target, class_index) in enumerate(dataloader):
        target = target.squeeze().long().to(device)
        inputs = inputs.to(device)
        # inference
        #outputs = model(inputs)
        # one-hot to index
        #pred = torch.argmax(outputs, dim=1)
        # to list
        probs = inputs[:, 0].detach().cpu().numpy()
        #preds = pred.detach().cpu().numpy()
        targets = target.detach().cpu().numpy()
        # print(preds)
        if prob_list is None:
            prob_list = probs
            #pred_list = preds
            true_list = targets
        else:
            prob_list = np.concatenate((prob_list, probs), axis=0)
            #pred_list = np.concatenate((pred_list, preds), axis=0)
            true_list = np.concatenate((true_list, targets), axis=0)
        torch.cuda.empty_cache()
    return prob_list, pred_list, true_list


def _mem_inf_roc(labels, results):
    """MIA AUC given the feature values (no need to threshold)"""
    auc = roc_auc_score(labels, results)
    # ap = average_precision_score(labels, results)
    info = 'the attack auc is {auc:.3f}'.format(auc=auc)
    print(info)
    return auc


@torch.no_grad()
def get_target_preds(model, dataloader):
    pred_list = None
    true_list = None
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    # for i, (inputs, target, group_index) in enumerate(dataloader):
    for i, (inputs, target) in enumerate(dataloader):
        target = target.squeeze().long().to(device)
        inputs = inputs.to(device)
        # inference
        outputs, fea = model(inputs)
        # one-hot to index
        pred = torch.argmax(outputs, dim=1)
        # to list
        preds = pred.detach().cpu().numpy()
        targets = target.detach().cpu().numpy()
        # print(preds)
        if pred_list is None:
            pred_list = preds
            true_list = targets
        else:
            pred_list = np.concatenate((pred_list, preds), axis=0)
            true_list = np.concatenate((true_list, targets), axis=0)
        torch.cuda.empty_cache()
    return pred_list, true_list


def run(args, run_idx):
    """"""
    ''' get model '''
    target_model, shadow_model_list, attack_model = get_models(args=args, run_idx=run_idx)
    ''' get data '''
    target_train_loader, target_test_loader, shadow_train_loader_list, shadow_test_loader_list = get_data(args=args)
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
        shadow_test_list=shadow_test_loader_list
    )
    mia_manager.produce_attack_test(args=args)
    # mia_manager.attack_train
    attack_dataloader = mia_manager.attack_test
    ''' get target preds '''
    target_train_pred, target_train_true = get_target_preds(target_model, target_train_loader)
    target_test_pred, target_test_true = get_target_preds(target_model, target_test_loader)
    target_train_pred = np.squeeze(target_train_pred)
    target_train_true = np.squeeze(target_train_true)
    target_test_pred = np.squeeze(target_test_pred)
    target_test_true = np.squeeze(target_test_true)
    ''' get attack preds '''
    attack_value, attack_pred, attack_true = get_attack_preds(attack_model, attack_dataloader)
    # attack_pred = np.squeeze(attack_pred)
    attack_true = np.squeeze(attack_true)
    ''' compute train & test acc, MIAs auc '''

    target_train_df = {
        f'acc': [],
    }
    target_train_df['acc'] += [accuracy_score(target_train_true, target_train_pred)]
    # target_train_df = pd.DataFrame(target_train_df)

    target_test_df = {
        'acc': [],
    }
    target_test_df['acc'] += [accuracy_score(target_test_true, target_test_pred)]
    # target_test_df = pd.DataFrame(target_test_df)

    attack_df = {
        'auc': [],
    }
    # attack_df['acc'] += [accuracy_score(attack_true, attack_pred)]
    if attack_method != 'vanilla':
        attack_df['auc'] += [_mem_inf_roc(1 - attack_true, attack_value)]
    else:
        attack_df['auc'] += [_mem_inf_roc(attack_true, attack_value)]
    # attack_df = pd.DataFrame(attack_df)
    return target_train_df, target_test_df, attack_df


def main(args, num_run):
    target_train = None
    target_test = None
    attack_acc_auc = None

    folder_name = 'csv_acc_auc'

    ''' each run '''
    for i in range(num_run):
        cur_run = i + 1
        save_path = f'{save_base_path}/{arch_conf}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = f'{save_path}/{learn_method}_{attack_method}_k{args.k}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = f'{save_path}/{folder_name}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = f'{save_path}/{loss_conf}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        target_train_df, target_test_df, attack_df = run(
            args, cur_run
        )
        try:
            if target_train is None:
                target_train = target_train_df
                target_test = target_test_df
                attack_acc_auc = attack_df
            else:
                target_train['acc'] += target_train_df['acc']
                target_test['acc'] += target_test_df['acc']
                #attack_acc_auc['acc'] += attack_df['acc']
                attack_acc_auc['auc'] += attack_df['auc']
        except:
            print(save_path)
            continue
    print(target_train)
    target_train = pd.DataFrame(target_train)
    save_all_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/{folder_name}/{loss_conf}/{arch_conf}_target_train.csv'
    target_train.to_csv(save_all_path)
    print(target_test)
    target_test = pd.DataFrame(target_test)
    save_all_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/{folder_name}/{loss_conf}/{arch_conf}_target_test.csv'
    target_test.to_csv(save_all_path)
    print(attack_acc_auc)
    attack_acc_auc = pd.DataFrame(attack_acc_auc)
    save_all_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/{folder_name}/{loss_conf}/{arch_conf}_attack_test.csv'
    attack_acc_auc.to_csv(save_all_path)
    # print(df)
    # print(df_all)
    # print(target_train_df)
    # print(target_train_df_all)
    # print(target_test_df)
    # print(target_test_df_all)


def get_models(args, run_idx):
    """"""
    ''' target model '''
    target_model = torch.load(
        f'{args.load_path}/{run_idx}/{args.arch}/{args.arch_conf}/target/model_latest_target.path.tar'
    )
    # ''' shadow models '''
    # shadow_model_list = [
    #    torch.load(
    #        f'{args.load_path}/{run_idx}/{args.arch}/{args.arch_conf}/shadow/model_latest_shadow_{idx}.path.tar'
    #    ) for idx in range(k)
    # ]
    # ''' get attack models '''
    # attack_model = torch.load(
    #    f'{args.base_path}/{run_idx}/{args.arch}/{args.arch_conf}/attack/model_latest_attack.path.tar'
    # )
    return target_model, [], None


def get_data(args):
    """"""
    from dataloader import data_selector
    batch_size = args.batch_size
    num_workers = args.workers

    the_dataset = data_selector.dataloader_selector(
        dataset=args.dataset, k=args.k, split_train=split_ratio
    )
    target_train_set = the_dataset['target_train']
    target_test_set = the_dataset['target_test']
    shadow_train_set_list = the_dataset['shadow_train']
    shadow_test_set_list = the_dataset['shadow_test']
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
    k = args.k
    ''' shadow datasets '''
    shadow_train_loader_list = []
    shadow_test_loader_list = []
    for i in range(k):
        shadow_train_set = shadow_train_set_list[i]
        shadow_test_set = shadow_test_set_list[i]

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
    return target_train_loader, target_test_loader, shadow_train_loader_list, shadow_test_loader_list


if __name__ == '__main__':
    print(torch.cuda.is_available())
    from conf.settings import *
    import torch.utils.data as data
    from dataloader import info

    ## dataset
    # dataset_name = 'cifar100'
    # dataset = f'{dataset_name}_test'
    # num_classes = info.num_classes[dataset_name]
    # num_group = 100
    ## model
    # arch = 'resnet_c'
    # arch_conf = 'resnet18_42'
    # k = 5
    # attack_method = 'vanilla'
    # learn_method = 'vanilla'
    # loss_conf = 'vanilla'
    # split_ratio = 0.0
    ## others
    num_runs = 3
    ## path
    # base_path = f'/mnt/ssd2/experiment/pfr/{loss_conf}/{dataset_name}/{arch_conf}/{learn_method}_{attack_method}'
    # save_base_path = f'/mnt/ssd2/result/pm/{dataset_name}/{arch}'
    # if not os.path.exists(f'/mnt/ssd2/result/pm/{dataset_name}'):
    #    os.mkdir(f'/mnt/ssd2/result/pm/{dataset_name}')
    # if not os.path.exists(save_base_path):
    #    os.mkdir(save_base_path)
    parser = argparse.ArgumentParser(description='PyTorch Eval')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--arch', type=str)
    parser.add_argument('--arch-conf', type=str)
    parser.add_argument('--learn-method', type=str)
    parser.add_argument('--attack-method', type=str)
    parser.add_argument('--loss-conf', type=str)
    parser.add_argument('--k', type=int)
    parser.add_argument('--split-ratio', type=float)
    # parser.add_argument('--base-path', type=str)

    args = parser.parse_args()
    # dataset
    batch_size = args.batch_size
    dataset_name = args.dataset
    dataset = dataset_name
    # dataset = f'{args.dataset}_test'
    num_classes = info.num_classes[dataset_name]
    num_group = num_classes
    # model
    arch = args.arch
    arch_conf = args.arch_conf
    k = args.k
    attack_method = args.attack_method
    learn_method = args.learn_method
    loss_conf = args.loss_conf
    split_ratio = args.split_ratio

    # base_path = f"/home/ubuntu/Data_Free_Privacy/save_checkpoints/{loss_conf}/{dataset_name}/{arch_conf}/{learn_method}_{attack_method}"
    # meta_path = f"./save_checkpoints/{loss_conf}/{dataset_name}/{arch_conf}"
    # meta_path = f"I:/mias/{loss_conf}/{dataset_name}/{arch_conf}"
    # meta_path = f"/home/xfang23/saves1/{loss_conf}/{dataset_name}/{arch_conf}"
    meta_path = f"/mnt/beegfs/xfang23/saves3/{loss_conf}/{dataset_name}/{arch_conf}"
    #meta_path = f"/media/xfang23/TOSHIBA EXT/mia2/{loss_conf}/{dataset_name}/{arch_conf}"
    # meta_path = f"/home/ubuntu/saves1/{loss_conf}/{dataset_name}/{arch_conf}"
    base_path = f"{meta_path}/{learn_method}_{attack_method}"
    # base_path = f'/mnt/ssd2/experiment/pfr/{loss_conf}/{dataset_name}/{arch_conf}/{learn_method}_{attack_method}'
    load_path = f"{meta_path}/{learn_method}_vanilla"

    args.base_path = base_path
    args.load_path = load_path
    # temp = f'/Volumes/T7/result/mias/csv'
    #temp = f'I:/result/mias/csv'
    #temp = f'/home/ubuntu/result/mias/csv'
    #temp = f'/home/xfang23/result/mias/csv'
    temp = f'/mnt/beegfs/xfang23/result/mias/csv'
    temp = temp
    if not os.path.exists(temp):
        os.mkdir(temp)
    temp = f'{temp}/acc_auc'
    if not os.path.exists(temp):
        os.mkdir(temp)
    temp = f'{temp}/{dataset_name}'
    if not os.path.exists(temp):
        os.mkdir(temp)
    temp = f'{temp}/{arch}'
    if not os.path.exists(temp):
        os.mkdir(temp)
    # save_base_path = f'/home/ubuntu/csv/acc_auc/{dataset_name}/{arch}'
    save_base_path = temp

    # print(args.base_path)
    args.dataset = dataset
    args.batch_size = batch_size
    args.device = 'cuda'
    args.if_pruning = 'no'
    args.workers = 4

    if not os.path.exists(save_base_path):
        os.mkdir(save_base_path)

    main(args, num_runs)
