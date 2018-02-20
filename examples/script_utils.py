#!/usr/bin/env python

import datetime
import os
import os.path as osp
import shlex
import subprocess

import pytz
import yaml

import torchfcn
import torch
from glob import glob

here = osp.dirname(osp.abspath(__file__))
MY_TIMEZONE = 'America/New_York'


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_latest_logdir_with_checkpoint(log_parent_directory):
    checkpoint_paths = glob(os.path.join(log_parent_directory, '*/checkpoint.pth.tar'))
    checkpoint_paths.sort(key=os.path.getmtime)
    return os.path.dirname(checkpoint_paths[-1])


def get_cityscapes_train_loader(root, n_max_per_class, resized_sz=None, set_extras_to_void=True,
                                semantic_subset=None, modified_length=None,
                                cuda=True, num_workers=4, pin_memory=True, batch_size=1,
                                **dataset_kwargs):
    # 1. dataset
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if cuda else {}
    train_dataset = torchfcn.datasets.CityscapesClassSegBase(root, split='train', transform=True,
                                                             semantic_subset=semantic_subset,
                                                             n_max_per_class=n_max_per_class,
                                                             resized_sz=resized_sz,
                                                             set_extras_to_void=set_extras_to_void,
                                                             **dataset_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, **loader_kwargs)
    if modified_length is not None:
        train_dataset_for_val = train_dataset.copy(modified_length=10)
        train_loader_for_val = torch.utils.data.DataLoader(train_dataset_for_val, batch_size=1,
                                                           shuffle=False)
        return train_loader_for_val
    return train_loader


def get_cityscapes_val_loader(root, n_max_per_class, resized_sz=None, set_extras_to_void=True,
                              semantic_subset=None, modified_length=None,
                              cuda=True, num_workers=4, pin_memory=True, batch_size=1,
                              **dataset_kwargs):
    # 1. dataset
    loader_kwargs = {'num_workers': num_workers, 'pin_memory': pin_memory} if cuda else {}
    train_dataset = torchfcn.datasets.CityscapesClassSegBase(root, split='val', transform=True,
                                                             semantic_subset=semantic_subset,
                                                             n_max_per_class=n_max_per_class,
                                                             resized_sz=resized_sz,
                                                             set_extras_to_void=set_extras_to_void,
                                                             **dataset_kwargs)
    val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                             shuffle=False, **loader_kwargs)
    if modified_length is not None:
        val_dataset_for_val = train_dataset.copy(modified_length=10)
        val_loader = torch.utils.data.DataLoader(val_dataset_for_val, batch_size=1,
                                                 shuffle=False)
    return val_loader


def get_latest_model_path_from_logdir(logdir):
    assert os.path.exists(logdir), '{} doesn\'t exist'.format(logdir)
    best_model_path = os.path.join(logdir, 'model_best.pth.tar')
    last_checkpoint_path = os.path.join(logdir, 'checkpoint.pth.tar')
    model_path = best_model_path if os.path.isfile(best_model_path) else None
    model_path = last_checkpoint_path if model_path is None and os.path.isfile(
        last_checkpoint_path) else model_path
    assert model_path is not None, 'Neither {} nor {} exists'.format(best_model_path,
                                                                     last_checkpoint_path)
    return model_path


def git_hash():
    cmd = 'git log -n 1 --pretty="%h"'
    hash = subprocess.check_output(shlex.split(cmd)).strip()
    return hash


def get_log_dir(model_name, config_id, cfg, parent_directory=None):
    # load config
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    for k, v in cfg.items():
        v = str(v)
        if '/' in v:
            continue
        name += '_%s-%s' % (k.upper(), v)
    now = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))
    name += '_VCS-%s' % git_hash()
    name += '_TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    # create out
    if parent_directory is None:
        parent_directory = here
    log_dir = osp.join(parent_directory, 'logs', name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
        torchfcn.models.FCN8sInstance,
    )
    for m in model.modules():
        # import ipdb; ipdb.set_trace()
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            import ipdb; ipdb.set_trace()
            raise ValueError('Unexpected module: %s' % str(m))
