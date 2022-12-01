# ------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# ------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# ------------------------------------------
"""
Train and eval functions used in main.py
"""
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path

import torch

import numpy as np

from timm.utils import accuracy
from timm.optim import create_optimizer
from continual_dataloader import get_train_loaders, get_val_loaders

import utils

def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, args=None, source_cov=None, proj=None, writer=None, cur_iter=0):

    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch+1:{int(math.log10(args.epochs))+1}}/{args.epochs}]'

    for input, target, _ in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
        
        output = model(input, task_id=task_id, cls_features=cls_features, train=set_training_mode)
        logits = output['logits']
        if task_id <= 4:
            # here is the trick to mask out classes of non-current tasks
            if args.train_mask and class_mask is not None:
                mask = class_mask[task_id]
                not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))

            target = target.type(torch.LongTensor).to(device)
            loss_cls = criterion(logits, target) # base criterion (CrossEntropyLoss)

        else:
            # entropy minimization (SHOT)
            probs = torch.nn.Softmax(dim=1)(logits)
            entropy_loss = torch.mean(torch.sum(-probs * torch.log(probs + 1e-7), dim=1))
            msoftmax = probs.mean(dim=0)
            gentropy_loss = torch.sum(-msoftmax * torch.log(msoftmax + 1e-7))
            entropy_loss -= gentropy_loss
            loss_cls = entropy_loss * 1.0

            # # coral aligning
            # pre_logits = output['pre_logits']
            # cur_cov = utils.compute_covariance(pre_logits)
            # loss_coral = torch.sum(torch.mul((source_cov - cur_cov), (source_cov - cur_cov)))
            # # loss += loss_coral / (4 * source_cov.shape[1] * source_cov.shape[1])
            # loss = loss_coral / (4 * source_cov.shape[1] * source_cov.shape[1])

        if args.pull_constraint and 'reduce_sim' in output:
            loss_sim = args.pull_constraint_coeff * output['reduce_sim']
            loss = loss_cls - loss_sim

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)
        writer.add_scalar("Loss/train_total_loss", loss, cur_iter)
        writer.add_scalar("Loss/train_cls_loss", loss_cls, cur_iter)
        writer.add_scalar("Loss/train_sim_loss", loss_sim, cur_iter)
        writer.add_scalar("Acc/train", acc1, cur_iter)
        cur_iter += 1
        optimizer.zero_grad()
        loss.backward()
        if proj is not None and len(proj) > 0:
            a = model.head.parameters()
            for p_w, p_b in zip(a, a):
                p_grad = torch.cat((p_w.grad, torch.unsqueeze(p_b.grad, 1)), dim=1)
                for subspace in proj:
                    p_grad = torch.mm(torch.mm(p_grad, subspace), subspace.t())
                p_w.grad = p_grad[:, :-1]
                p_b.grad = p_grad[:, -1]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, cur_iter

@torch.no_grad()
def calculate_cov(model: torch.nn.Module, original_model: torch.nn.Module, data_loader: Iterable,
                    device: torch.device, task_id=-1, args=None):
    model.eval()
    original_model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Calculate Covariance'

    # source_cov = torch.zeros((768, 768)).to(args.device)
    feat_mtx = torch.zeros((len(data_loader.dataset), 768)).to(args.device)
    idx = 0
    for input, _, _  in data_loader:
        input = input.to(device, non_blocking=True)
        # target = target.to(device, non_blocking=True)

        with torch.no_grad():
            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

        output = model(input, task_id=task_id, cls_features=cls_features, train=False)

        pre_logits = output['pre_logits']
        # source_cov += utils.compute_covariance(pre_logits)
        feat_mtx[idx:idx+input.shape[0]] = pre_logits
        idx += input.shape[0]
    # print('************ Source Covariance Matrix **************')
    # print(source_cov)
    feat_mtx = torch.cat((feat_mtx, torch.ones((feat_mtx.shape[0], 1)).to(args.device)), dim=1)
    mtx = torch.mm(feat_mtx.t(), feat_mtx) / feat_mtx.shape[0]
    # return source_cov / len(data_loader)
    return mtx


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
            device, task_id=-1, class_mask=None, args=None,):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None
            
            output = model(input, task_id=task_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                #adding mask to output logits
                mask = class_mask[task_id]
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'], losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_continuum(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, task_id=-1, class_mask=None, args=None, ):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(task_id + 1)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    with torch.no_grad():
        for input, target, _ in metric_logger.log_every(data_loader, args.print_freq, header):
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output

            if original_model is not None:
                output = original_model(input)
                cls_features = output['pre_logits']
            else:
                cls_features = None

            # output = model(input, task_id=task_id, cls_features=cls_features)
            # prompt_id = 0
            prompt_id = task_id
            output = model(input, task_id=prompt_id, cls_features=cls_features)
            logits = output['logits']

            if args.task_inc and class_mask is not None:
                # adding mask to output logits
                mask = class_mask[task_id]
                logits_mask = torch.ones_like(logits, device=device) * float('-inf')
                logits_mask = logits_mask.index_fill(1, mask, 0.0)
                logits = logits + logits_mask

            target = target.type(torch.LongTensor).to(device)
            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                  losses=metric_logger.meters['Loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader, 
                    device, task_id=-1, class_mask=None, acc_matrix=None, args=None,):
    stat_matrix = np.zeros((3, args.num_tasks)) # 3 for Acc@1, Acc@5, Loss

    for i in range(task_id+1):
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'], 
                            device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']
    
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id+1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id+1, avg_stat[0], avg_stat[1], avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                            acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


@torch.no_grad()
def evaluate_till_now_continuum(model: torch.nn.Module, original_model: torch.nn.Module, scenario_val,
                      device, task_id=-1, class_mask=None, acc_matrix=None, args=None, ):
    # stat_matrix = np.zeros((3, args.num_tasks))  # 3 for Acc@1, Acc@5, Loss
    stat_matrix = np.zeros((3, len(scenario_val)))  # 3 for Acc@1, Acc@5, Loss
    # stat_matrix = np.zeros((3, task_id+1))  # 3 for Acc@1, Acc@5, Loss

    for i, dataset_val in enumerate(scenario_val):
        # if i > task_id:
        #     break
        loader_val = get_val_loaders(dataset_val, args)
        test_stats = evaluate_continuum(model=model, original_model=original_model, data_loader=loader_val,
                              device=device, task_id=i, class_mask=class_mask, args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']

        acc_matrix[i, task_id] = test_stats['Acc@1']

    # avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)
    avg_stat = np.divide(np.sum(stat_matrix, axis=1), len(scenario_val))
    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(task_id + 1,
                                                                                                     avg_stat[0],
                                                                                                     avg_stat[1],
                                                                                                     avg_stat[2])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats, stat_matrix


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module, 
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer, lr_scheduler, device: torch.device, 
                    class_mask=None, args=None,):

    # create matrix to save end-of-task accuracies 
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

    for task_id in range(args.num_tasks):
        if task_id < 2:
            continue
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        
        for epoch in range(args.epochs):            
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion, 
                                        data_loader=data_loader[task_id]['train'], optimizer=optimizer, 
                                        device=device, epoch=epoch, max_norm=args.clip_grad, 
                                        set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args,)
            
            if lr_scheduler:
                lr_scheduler.step(epoch)

        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader, device=device, 
                                    task_id=task_id, class_mask=class_mask, acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)
            
            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()
            
            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
            **{f'test_{k}': v for k, v in test_stats.items()},
            'epoch': epoch,}

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir, '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))), 'a') as f:
                f.write(json.dumps(log_stats) + '\n')


def train_and_evaluate_continuum(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, scenario_train, scenario_val, optimizer: torch.optim.Optimizer, lr_scheduler,
                       device: torch.device,
                       class_mask=None, args=None, writer=None):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    source_cov = None
    projection = []
    cur_iter = 0
    for task_id, dataset_train in enumerate(scenario_train):
    # for task_id in [7]:
    #     Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
        loader_train = get_train_loaders(dataset_train, args)

        # if task_id == 0:
        #     source_cov = calculate_cov(model=model, original_model=original_model, data_loader=loader_train,
        #                                 device=device, set_training_mode=True, task_id=task_id, args=args)
        #     continue
        # if task_id <=1:
        #     print('Skip {}-th task!'.format(task_id + 1))
        #     continue
        for epoch in range(args.epochs):
            train_stats, cur_iter = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                          data_loader=loader_train, optimizer=optimizer,
                                          device=device, epoch=epoch, max_norm=args.clip_grad,
                                          set_training_mode=True, task_id=task_id, class_mask=class_mask, args=args, source_cov=source_cov, proj=projection, writer=writer, cur_iter=cur_iter)

            if lr_scheduler:
                lr_scheduler.step(epoch)

            test_stats, stat_matrix = evaluate_till_now_continuum(model=model, original_model=original_model, scenario_val=scenario_val,
                                                     device=device, task_id=task_id, class_mask=class_mask,
                                                     acc_matrix=acc_matrix, args=args)
            for task_id_val in range(len(scenario_val)):
                writer.add_scalar("Acc/Val_task{}".format(task_id_val), stat_matrix[0, task_id_val], cur_iter)
                writer.add_scalar("Loss/Val_task{}".format(task_id_val), stat_matrix[2, task_id_val], cur_iter)
            if args.output_dir and utils.is_main_process():
                Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

                checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint{}.pth'.format(task_id + 1, epoch))
                state_dict = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }
                if args.sched is not None and args.sched != 'constant':
                    state_dict['lr_scheduler'] = lr_scheduler.state_dict()

                utils.save_on_master(state_dict, checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'stat': stat_matrix.tolist()}

            if args.output_dir and utils.is_main_process():
                with open(os.path.join(args.output_dir,
                                       '{}_stats_epoch{}.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'), epoch)),
                          'a') as f:
                    f.write(json.dumps(log_stats) + '\n')

        if args.orthogonal_head:
            feature_mtx = calculate_cov(model=model, original_model=original_model, data_loader=loader_train,
                                            device=device, task_id=task_id, args=args)
            u, s, v = torch.svd(feature_mtx)
            # print('****U****')
            # print(u)
            # print('****S****')
            # print(s)
            # print(s<torch.min(s)*1000)
            # print('****V****')
            # print(v)
            # v[:, :180] = 0
            # print(torch.mm(feature_mtx, v))
            projection.append(v[:, 500:])


@torch.no_grad()
def analyze(model: torch.nn.Module, original_model: torch.nn.Module, scenario,
             device, task_id=-1, class_mask=None, args=None, ):

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    p_stat = {'class': np.zeros((args.nb_classes, args.size)), 'task': np.zeros((args.num_tasks, args.size))}

    with torch.no_grad():
        for task_id, dataset in enumerate(scenario):
            data_loader = get_train_loaders(dataset, args)
            header = 'Test: [Task {}]'.format(task_id + 1)
            for input, target, _ in metric_logger.log_every(data_loader, args.print_freq, header):
                input = input.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # compute output

                if original_model is not None:
                    output = original_model(input)
                    cls_features = output['pre_logits']
                else:
                    cls_features = None

                output = model(input, task_id=task_id, cls_features=cls_features, return_prompt=True)
                for i in range(target.shape[0]):
                    p_stat['task'][task_id][output['idx'][i].cpu()] += 1
                    p_stat['class'][target[i]][output['idx'][i].cpu()] += 1
            print(p_stat)
    return p_stat

@torch.no_grad()
def draw_tsne(model, original_model, scenario, device, args=None):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    metric_logger = utils.MetricLogger(delimiter="  ")
    model.eval()
    original_model.eval()
    with torch.no_grad():
        for task_id, dataset in enumerate(scenario):
            for prompt in range(3):
                feature_matrix = np.zeros((len(dataset), 768))
                target_arr = np.zeros(len(dataset))
                cur_idx = 0
                data_loader = get_train_loaders(dataset, args)
                header = 'Test: [Task {}]'.format(task_id + 1)
                dom_id = task_id if prompt == 0 else 3 + prompt
                dom_arr = np.ones(len(dataset)) * dom_id
                if task_id >= 1 and prompt > 0:
                    continue
                for input, target, _ in metric_logger.log_every(data_loader, args.print_freq, header):
                    input = input.to(device, non_blocking=True)
                    target = target
                    if original_model is not None:
                        output = original_model(input, task_id=0)
                        cls_features = output['pre_logits']
                    else:
                        cls_features = None

                    output = model(input, task_id=prompt, cls_features=cls_features)
                    feature_matrix[cur_idx:cur_idx+input.shape[0]] = output['pre_logits'].cpu()
                    target_arr[cur_idx:cur_idx+input.shape[0]] = target
                    cur_idx += input.shape[0]
                if task_id == 0 and prompt == 0:
                    feats = feature_matrix[target_arr<=6]
                    tgts = target_arr[target_arr<=6]
                    doms = dom_arr[target_arr<=6]
                else:
                    feats = np.concatenate((feats, feature_matrix[target_arr<=6]), axis=0)
                    tgts = np.concatenate((tgts, target_arr[target_arr<=6]), axis=0)
                    doms = np.concatenate((doms, dom_arr[target_arr<=6]), axis=0)
                # selected_idx = np.random.choice(len(dataset), len(dataset)//10, replace=False)
                # tsne_np = TSNE(n_components=2).fit_transform(feature_matrix[selected_idx])
                # tsne_np = TSNE(n_components=2).fit_transform(feature_matrix[target_arr<10])
                tsne_np = TSNE(n_components=2).fit_transform(feats)
                # selected_target = target_arr[selected_idx]
                # selected_target = target_arr[target_arr<10]
                selected_target = tgts
                plt.figure(figsize=(20,20))
                # for cls in range(345):
                shape = ['o', 'v', 'x', 's', '+', 'd']
                colors = ['red', 'orange', 'blue', 'green', 'purple', 'pink', 'gray', 'cyan', 'black', 'olive']
                for _dom in range(5):
                    dom = _dom + 1
                    for _cls in range(6):
                        cls = _cls
                        # color = np.array([list(np.random.choice(range(256), size=3))] * (selected_target==cls).sum())
                        plt.scatter(tsne_np[(selected_target==cls)&(doms==dom)][:,0], tsne_np[(selected_target==cls)&(doms==dom)][:,1], s=30, c=colors[_cls], marker=shape[dom], label='{}'.format(cls+1))
                plt.xlabel('component 0')
                plt.ylabel('component 1')
                # plt.legend()
                plt.savefig(os.path.join(args.output_dir, 'tsne_task{}_from_dom2.jpg'.format(task_id+1)))
                plt.figure(figsize=(20,20))
                for dom in range(6):
                    plt.scatter(tsne_np[doms==dom][:,0], tsne_np[doms==dom][:,1], s=30, c=colors[dom], marker=shape[dom], label='{}'.format(cls+1))
                plt.savefig(os.path.join(args.output_dir, 'tsne_task{}_by_domain.jpg'.format(task_id + 1)))
    return
