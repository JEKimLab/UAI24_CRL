import argparse

import pandas as pd
import torch
from summarization import load_model
from summarization.metric_generator import MetricGenerator

batch_size = 256
num_workers = 4

device = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def get_attack_preds_in_group(model, dataloader):
    groups_pred = [[] for _ in range(num_group)]
    groups_true = [[] for _ in range(num_group)]
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    for i, (inputs, target, group_index) in enumerate(dataloader):
        if len(inputs.size()) == 1:
            inputs = torch.unsqueeze(inputs, 1)
        target = target.squeeze().long().to(device)
        inputs = inputs.to(device)
        # inference
        outputs = model(inputs)
        # one-hot to index
        pred = torch.argmax(outputs, dim=1)
        # to list
        preds = list(pred.cpu())
        targets = list(target.cpu())
        group_indexs = list(group_index.cpu())
        for idx in range(len(group_indexs)):
            group_idx = group_indexs[idx]
            groups_pred[group_idx] += [preds[idx]]
            groups_true[group_idx] += [targets[idx]]
        torch.cuda.empty_cache()
    return groups_pred, groups_true


@torch.no_grad()
def get_attack_preds(model, dataloader):
    pred_list = []
    true_list = []
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    for i, (inputs, target, group_index) in enumerate(dataloader):
        target = target.squeeze().long().to(device)
        inputs = inputs.to(device)
        # inference
        outputs = model(inputs)
        # one-hot to index
        pred = torch.argmax(outputs, dim=1)
        # to list
        preds = list(pred.cpu())
        targets = list(target.cpu())
        # print(preds)
        pred_list += preds
        true_list += targets
        torch.cuda.empty_cache()
    return pred_list, true_list


@torch.no_grad()
def get_predictions(model, dataloader):
    preds = []
    targets = []
    groups = []
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    model.eval()
    for i, (inputs, target, group_index) in enumerate(dataloader):
        # inference
        target = target.squeeze().long().to(device)
        inputs = inputs.to(device)
        output, fea = model(inputs)
        ''' softmax could be important '''
        output = torch.softmax(output, dim=-1)
        # add to list
        preds += list(output.cpu())
        targets += list(target.cpu())
        groups += list(group_index.cpu())
        torch.cuda.empty_cache()
    return preds, targets, groups


@torch.no_grad()
def get_target_preds(model, dataloader):
    pred_list = []
    true_list = []
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    for i, (inputs, target, group_index) in enumerate(dataloader):
        target = target.squeeze().long().to(device)
        inputs = inputs.to(device)
        # inference
        outputs, fea = model(inputs)
        # one-hot to index
        pred = torch.argmax(outputs, dim=1)
        # to list
        preds = list(pred.cpu())
        targets = list(target.cpu())
        # print(preds)
        pred_list += preds
        true_list += targets
        torch.cuda.empty_cache()
    return pred_list, true_list


@torch.no_grad()
def get_target_preds_in_group(model, dataloader):
    groups_pred = [[] for _ in range(num_group)]
    groups_true = [[] for _ in range(num_group)]
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    for i, (inputs, target, group_index) in enumerate(dataloader):
        target = target.squeeze().long().to(device)
        inputs = inputs.to(device)
        # inference
        outputs, fea = model(inputs)
        # one-hot to index
        pred = torch.argmax(outputs, dim=1)
        # to list
        preds = list(pred.cpu())
        targets = list(target.cpu())
        group_indexs = list(group_index.cpu())
        for idx in range(len(group_indexs)):
            group_idx = group_indexs[idx]
            groups_pred[group_idx] += [preds[idx]]
            groups_true[group_idx] += [targets[idx]]
        torch.cuda.empty_cache()
    return groups_pred, groups_true

@torch.no_grad()
def get_target_dist(model, dataloader):
    pred_list = []
    true_list = []
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
    for i, (inputs, target, group_index) in enumerate(dataloader):
        target = target.squeeze().long().to(device)
        inputs = inputs.to(device)
        with torch.no_grad():
            logits, fea = model(inputs)
        # inference
        dist = distance_to_decision_boundary(model, logits)
        # one-hot to index
        #pred = torch.argmax(outputs, dim=1)
        # to list
        preds = list(dist.cpu())
        targets = list(target.cpu())
        # print(preds)
        pred_list += preds
        true_list += targets
        torch.cuda.empty_cache()
    return pred_list, true_list

@torch.no_grad()
def get_target_dist_in_group(model, attack_model, dataloader, mode='train'):
    groups_pred = [[] for _ in range(num_group)]
    groups_true = [[] for _ in range(num_group)]
    groups_pred_0 = [[] for _ in range(num_group)]
    group_attack_pred = [[] for _ in range(num_group)]
    group_attack_true = [[] for _ in range(num_group)]
    if device == 'cuda':
        model = torch.nn.DataParallel(model).to(device)
        attack_model = attack_model.to(device)
    for i, (inputs, target, group_index) in enumerate(dataloader):
        target = target.squeeze().long().to(device)
        inputs = inputs.to(device)
        # inference
        with torch.no_grad():
            logits, fea = model(inputs)
        dist = distance_to_decision_boundary(model, logits)
        dist_0 = distance_to_zero(model, fea)
        mia_pred, mia_true = mia_infer(model, attack_model, logits, target, mode)
        # one-hot to index
        mia_pred = torch.argmax(mia_pred, dim=1)
        # to list
        preds = list(dist.cpu())
        preds_0 = list(dist_0.cpu())
        targets = list(target.cpu())
        mia_pred = list(mia_pred.cpu())
        mia_true = list(mia_true.cpu())
        group_indexs = list(group_index.cpu())
        for idx in range(len(group_indexs)):
            group_idx = group_indexs[idx]
            groups_pred[group_idx] += [preds[idx]]
            groups_pred_0[group_idx] += [preds_0[idx]]
            groups_true[group_idx] += [targets[idx]]
            group_attack_pred[group_idx] += [mia_pred[idx]]
            group_attack_true[group_idx] += [mia_true[idx]]
        torch.cuda.empty_cache()
    return groups_pred, groups_true, groups_pred_0, group_attack_pred, group_attack_true

def mia_infer(model, attack_model, logits, y, mode='train'):
    #x.requires_grad_(False)
    #logits, fea = model(x)
    logits = torch.softmax(logits, dim=1)
    y = one_hot_embedding(y, num_classes=num_classes)
    inputs = torch.cat([logits, y], dim=1)
    attack_pred = attack_model(inputs)
    b, d = attack_pred.size()
    if mode=='train':
        attack_true = torch.ones(b, 1)
    else:
        attack_true = torch.zeros(b, 1)
    return attack_pred, attack_true

def one_hot_embedding(y, num_classes=10, dtype=torch.cuda.FloatTensor):
    """
    derived from: https://github.com/DingfanChen/RelaxLoss/blob/main/source/utils/ops.py
    apply one hot encoding on labels
    :param y: class label
    :param num_classes: number of classes
    :param dtype: data type
    :return:
    """
    scatter_dim = len(y.size())
    # y_tensor = y.type(torch.cuda.LongTensor).view(*y.size(), -1)
    y_tensor = y.view(*y.size(), -1)
    zeros = torch.zeros(*y.size(), num_classes).type(dtype)
    return zeros.scatter(scatter_dim, y_tensor, 1)

def distance_to_zero(model, logits):
    #x.requires_grad_(False)
    #logits, fea = model(x)
    with torch.no_grad():
        distance = torch.norm(logits, p=2, dim=1, keepdim=False)
    #print(distance)
    #x.requires_grad_(True)
    return distance

def distance_to_decision_boundary(model, logits):
    #x.requires_grad_(True)
    #logits, fea = model(x)
    # Reuse the computed activations and disable gradient tracking
    with torch.no_grad():
        logits = torch.softmax(logits, dim=-1)
        margin = torch.abs(
            torch.max(logits, dim=1, keepdim=True).values - logits
        )
        margin = torch.kthvalue(margin, k=2, dim=1).values
        distance = margin
    #print(distance)
    #x.requires_grad_(False)
    return distance


def compute_scores():
    pass


def save_to_csv():
    pass


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
    target_train_pred, target_train_true, target_train_pred_0, mia_train_pred, mia_train_true = get_target_dist_in_group(target_model, attack_model, target_train_loader, mode='train')
    target_test_pred, target_test_true, target_test_pred_0, mia_test_pred, mia_test_true = get_target_dist_in_group(target_model, attack_model, target_test_loader, mode='test')
    ''' get attack preds '''
    #groups_pred, groups_true = get_attack_preds_in_group(attack_model, attack_dataloader)
    ''' target train acc '''
    target_train_df_all = {
        'dist_to_b': [],
        'dist_to_o': [],
        'mia':[],
    }
    target_train_df = []
    for i in range(num_group):
        preds = target_train_pred[i]
        preds_0 = target_train_pred_0[i]
        trues = target_train_true[i]
        mia = mia_train_pred[i]
        mia_true = mia_train_true[i]
        #mg = MetricGenerator()
        #mg.update(preds, trues)
        target_train_df += [pd.DataFrame({
            'dist_to_b': [float(d) for d in preds],
            'dist_to_o': [float(d) for d in preds_0],
            'mia':[float(mia[d]==mia_true[d]) for d in range(len(mia))],
        })]
        target_train_df_all['dist_to_b'] += [float(d) for d in preds]
        target_train_df_all['dist_to_o'] += [float(d) for d in preds_0]
        target_train_df_all['mia'] += [float(mia[d]==mia_true[d]) for d in range(len(mia))]
    #preds, trues = get_target_dist(target_model, target_train_loader)
    target_train_df_all = pd.DataFrame(target_train_df_all)
    ''' target test acc '''
    target_test_df_all = {
        'dist_to_b': [],
        'dist_to_o': [],
        'mia': [],
    }
    target_test_df = []
    for i in range(num_group):
        preds = target_test_pred[i]
        preds_0 = target_test_pred_0[i]
        trues = target_test_true[i]
        mia = mia_test_pred[i]
        mia_true = mia_test_true[i]
        #mg = MetricGenerator()
        #mg.update(preds, trues)
        target_test_df += [pd.DataFrame({
            'dist_to_b': [float(d) for d in preds],
            'dist_to_o': [float(d) for d in preds_0],
            'mia': [float(mia[d] == mia_true[d]) for d in range(len(mia))],
        })]
        target_test_df_all['dist_to_b'] += [float(d) for d in preds]
        target_test_df_all['dist_to_o'] += [float(d) for d in preds_0]
        target_test_df_all['mia'] += [float(mia[d] == mia_true[d]) for d in range(len(mia))]
    #preds, trues = get_target_dist(target_model, target_test_loader)
    #mg = MetricGenerator()
    #mg.update(preds, trues)
    target_test_df_all = pd.DataFrame(target_test_df_all)
    ''' attack acc in group '''
    metrics_list = []
    #for i in range(num_group):
    #    preds = groups_pred[i]
    #    trues = groups_true[i]
    #    #mg = MetricGenerator()
    #    #mg.update(preds, trues)
    #    metrics_list += [list(float(preds))]
    df = pd.DataFrame(metrics_list)
    ''' attack acc in total '''
    preds, trues = get_attack_preds(attack_model, attack_dataloader)
    mg = MetricGenerator()
    mg.update(preds, trues)
    df_all = pd.DataFrame([mg.get_metrics()])
    return df, df_all, target_train_df, target_train_df_all, target_test_df, target_test_df_all


def main(args, num_run):
    df_list = []
    df_all_list = []
    target_train_df_list = []
    target_train_df_all_list = []
    target_test_df_list = []
    target_test_df_all_list = []
    ''' each run '''
    for i in range(num_run):
        cur_run = i + 1
        save_path = f'{save_base_path}/{arch_conf}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = f'{save_path}/{learn_method}_{attack_method}_k{args.k}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = f'{save_path}/csv_dist_zero_para'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        save_path = f'{save_path}/{loss_conf}'
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        df, df_all, target_train_df, target_train_df_all, target_test_df, target_test_df_all = run(
            args, cur_run
        )
        try:
            # df, df_all, target_train_df, target_train_df_all, target_test_df, target_test_df_all = run(
            #    floder_path, save_path, arch, arch_conf, if_pruned
            # )
            # print(df_all)
            df_list += [df]
            df_all_list += [df_all]
            target_train_df_list += [target_train_df]
            target_train_df_all_list += [target_train_df_all]
            target_test_df_list += [target_test_df]
            target_test_df_all_list += [target_test_df_all]
        except:
            print(save_path)
            continue

    df = df_list[0]
    df_all = df_all_list[0]
    target_train_df = target_train_df_list[0]
    target_train_df_all = target_train_df_all_list[0]
    target_test_df = target_test_df_list[0]
    target_test_df_all = target_test_df_all_list[0]
    for i in range(1, len(df_list)):
        df += df_list[i]
        df_all += df_all_list[i]
        for g_i in range(num_group):
            target_train_df[g_i] += target_train_df_list[i][g_i]
            target_test_df[g_i] += target_test_df_list[i][g_i]
        target_train_df_all += target_train_df_all_list[i]
        target_test_df_all += target_test_df_all_list[i]
    df = df / len(df_list)
    df_all = df_all / len(df_list)
    for g_i in range(num_group):
        target_train_df[g_i] /= len(target_train_df_list)
        target_test_df[g_i] /= len(target_test_df_list)
    target_train_df_all /= len(target_train_df_all_list)
    target_test_df_all /= len(target_test_df_all_list)
    #save_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/csv_dist/{arch_conf}_attack_gruop.csv'
    #save_all_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/csv_dist/{arch_conf}_attack.csv'
    #df.to_csv(save_path)
    #df_all.to_csv(save_all_path)
    for i in range(num_group):
        save_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/csv_dist_zero_para/{loss_conf}/{arch_conf}_target_train_gruop_{i}.csv'
        target_train_df[i].to_csv(save_path)

    save_all_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/csv_dist_zero_para/{loss_conf}/{arch_conf}_target_train.csv'
    target_train_df_all.to_csv(save_all_path)

    for i in range(num_group):
        save_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/csv_dist_zero_para/{loss_conf}/{arch_conf}_target_test_gruop_{i}.csv'
        target_test_df[i].to_csv(save_path)
    #save_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/csv_dist/{arch_conf}_target_test_gruop.csv'
    save_all_path = f'{save_base_path}/{arch_conf}/{learn_method}_{attack_method}_k{args.k}/csv_dist_zero_para/{loss_conf}/{arch_conf}_target_test.csv'
    #target_test_df.to_csv(save_path)
    target_test_df_all.to_csv(save_all_path)
    #print(df)
    #print(df_all)
    #print(target_train_df)
    #print(target_train_df_all)
    #print(target_test_df)
    #print(target_test_df_all)


def get_models(args, run_idx):
    """"""
    ''' target model '''
    target_model = torch.load(
        f'{args.base_path}/{run_idx}/{args.arch}/{args.arch_conf}/target/model_latest_target.path.tar'
    )
    ''' shadow models '''
    shadow_model_list = []
    #shadow_model_list = [
    #    torch.load(
    #        f'{args.base_path}/{run_idx}/{args.arch}/{args.arch_conf}/shadow/model_latest_shadow_{idx}.path.tar'
    #    ) for idx in range(k)
    #]
    ''' get attack models '''
    attack_model = torch.load(
        f'{args.base_path}/{run_idx}/{args.arch}/{args.arch_conf}/attack/model_latest_attack.path.tar'
    )
    return target_model, shadow_model_list, attack_model


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

    # dataset
    dataset_name = 'cifar100'
    dataset = f'{dataset_name}_test'
    num_classes = info.num_classes[dataset_name]
    num_group = 100
    batch_size = 512
    # model
    arch = 'resnet'
    arch_conf = 'resnet18'
    k = 5
    attack_method = 'vanilla'
    learn_method = 'crl'
    loss_conf = 'crl_resnet_c100_10'
    split_ratio = 0.0
    # others
    num_runs = 30
    # path
    base_path = f'/mnt/ssd2/experiment/pfr/{loss_conf}/{dataset_name}/{arch_conf}/{learn_method}_{attack_method}'
    save_base_path = f'/mnt/ssd2/result/pfr/{dataset_name}'

    parser = argparse.ArgumentParser(description='PyTorch training')
    args = parser.parse_args()

    args.dataset = dataset
    args.arch = arch
    args.arch_conf = arch_conf
    args.learn_method = learn_method
    args.loss_conf = loss_conf
    args.k = k
    args.base_path = base_path
    args.attack_method = attack_method
    args.learn_method = learn_method
    args.batch_size = batch_size
    args.device = 'cuda'
    args.workers = 4
    args.if_pruning = 'no'

    if not os.path.exists(save_base_path):
        os.mkdir(save_base_path)

    main(args, num_runs)
