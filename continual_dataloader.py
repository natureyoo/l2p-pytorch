# --------------------------------------------------------
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
# Modification:
# Added code for l2p implementation
# -- Jaeho Lee, dlwogh9344@khu.ac.kr
# --------------------------------------------------------
import os
import random

import torch
from torch.utils.data.dataset import Subset
from torchvision import datasets, transforms
from torchvision.transforms.transforms import Lambda
from continuum.datasets import Core50
from continuum import InstanceIncremental

import utils


class ContinualDataLoader:
    def __init__(self, args):
        self.args = args
        if not os.path.exists(self.args.data_path):
            os.makedirs(self.args.data_path)
        self.transform_train = build_transform(True, self.args)
        self.transform_val = build_transform(False, self.args)
        self._get_dataset(self.args.dataset)

    def _get_dataset(self, name):
        if name == 'CIFAR100':
            root = self.args.data_path
            self.dataset_train = datasets.CIFAR100(root=root, train = True, download = True, transform = self.transform_train)
            self.dataset_val = datasets.CIFAR100(root =root, train = False, transform = self.transform_val)
            self.args.nb_classes = 100
        elif name == 'Core50':
            root = self.args.data_path
            # self.dataset_train = datasets.ImageFolder(root=root, train=True, transform=self.transform_train)
            # self.dataset_val = datasets.ImageFolder(root=root, train=False, transform=self.transform_train)
            self.dataset_train = Core50(root, scenario='domains', classification='object', train=True)
            self.dataset_test = Core50(root, scenario='domains', classification='object', train=False)
            self.args.nb_classes = 50
        else:
            raise NotImplementedError(f"Not supported dataset: {self.args.dataset}")
        
    def create_dataloader(self):
        if self.args.dataset == 'CIFAR100':
            dataloader, class_mask = self.cifar_split()
        else:
            dataloader, class_mask = self.core50_split()
        
        return dataloader, class_mask
    
    def target_transform(self, x):
        # Target transform form splited dataset, 0~9 -> 0~9, 10~19 -> 0~9, 20~29 -> 0~9..
        return x - 10*(x//10)

    def cifar_split(self):
        dataloader = []
        labels = [i for i in range(self.args.nb_classes)] # [0, 1, 2, ..., 99]
        
        if self.args.shuffle:
            random.shuffle(labels)
        
        class_mask = list() if self.args.task_inc or self.args.train_mask else None
        
        for _ in range(self.args.num_tasks):
            train_split_indices = []
            test_split_indices = []
            
            scope = labels[:self.args.classes_per_task]
            labels = labels[self.args.classes_per_task:]
            
            if class_mask is not None:
                class_mask.append(scope)

            for k in range(len(self.dataset_train.targets)):
                if int(self.dataset_train.targets[k]) in scope:
                    train_split_indices.append(k)
                    
            for h in range(len(self.dataset_val.targets)):
                if int(self.dataset_val.targets[h]) in scope:
                    test_split_indices.append(h)
            
            # self.dataset_train.target_transform = Lambda(self.target_transform)
            # self.dataset_val.target_transform = Lambda(self.target_transform)

            dataset_train, dataset_val =  Subset(self.dataset_train, train_split_indices), Subset(self.dataset_val, test_split_indices)

            if self.args.distributed and utils.get_world_size() > 1:
                num_tasks = utils.get_world_size()
                global_rank = utils.get_rank()

                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True)
                
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            else:
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
            )

            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.pin_mem,
            )

            dataloader.append({'train': data_loader_train, 'val': data_loader_val})
        
        return dataloader, class_mask


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        scale = (0.08, 1.0)
        ratio = (3. / 4., 4. / 3.)
        transform = transforms.Compose([
            transforms.RandomResizedCrop(args.input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ])
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))
    t.append(transforms.ToTensor())
    
    return transforms.Compose(t)


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    # if args.data_set.lower() == 'cifar':
    #     dataset = CIFAR100(args.data_path, train=is_train, download=True)
    # elif args.data_set.lower() == 'imagenet100':
    #     dataset = ImageNet100(
    #         args.data_path, train=is_train,
    #         data_subset=os.path.join('./imagenet100_splits', "train_100.txt" if is_train else "val_100.txt")
    #     )
    # elif args.data_set.lower() == 'imagenet1000':
    #     dataset = ImageNet1000(args.data_path, train=is_train)
    if args.dataset.lower() == 'core50':
        dataset = Core50(args.data_path, scenario='domains', classification='object', train=is_train)
    else:
        raise ValueError(f'Unknown dataset {args.data_set}.')

    scenario = InstanceIncremental(
        dataset,
        transformations=transform.transforms
    )
    nb_classes = scenario.nb_classes

    return scenario, nb_classes


def get_sampler(dataset_train, dataset_val, args):
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        if args.repeated_aug:
            sampler_train = RASamplerNoDist(dataset_train, num_replicas=2, shuffle=True)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    return sampler_train, sampler_val


def get_train_sampler(dataset_train, args):
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        # if args.repeated_aug:
        #     sampler_train = RASampler(
        #         dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        #     )
        # else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    return sampler_train


def get_val_sampler(dataset_val, args):
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        # if args.dist_eval:
        #     if len(dataset_val) % num_tasks != 0:
        #         print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
        #               'This will slightly alter validation results as extra duplicate entries are added to achieve '
        #               'equal num of samples per-process.')
        #     sampler_val = torch.utils.data.DistributedSampler(
        #         dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        # else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    return sampler_val


def get_loaders(dataset_train, dataset_val, args, finetuning=False):
    sampler_train, sampler_val = get_sampler(dataset_train, dataset_val, args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=None if (finetuning and args.ft_no_sampling) else sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=len(sampler_train) > args.batch_size,
    )

    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    return loader_train, loader_val


def get_train_loaders(dataset_train, args, batch_size=None, drop_last=True):
    batch_size = batch_size or args.batch_size

    sampler_train = get_train_sampler(dataset_train, args)

    loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=drop_last,
    )
    return loader_train


def get_val_loaders(dataset_val, args, finetuning=False):
    sampler_val = get_val_sampler(dataset_val, args)
    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )
    return loader_val
