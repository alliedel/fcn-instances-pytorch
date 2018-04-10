#!/usr/bin/env python

import datetime
import os
import os.path as osp
import shlex
import subprocess

import pytz
import yaml

import torchfcn
from glob import glob

here = osp.dirname(osp.abspath(__file__))
MY_TIMEZONE = 'America/New_York'

CONFIG_KEY_REPLACEMENTS_FOR_FILENAME = {'max_iteration': 'itr',
                                        'weight_decay': 'decay',
                                        'n_training_imgs': 'n_train',
                                        'n_validation_imgs': 'n_val',
                                        'recompute_optimal_loss': 'recomp',
                                        'size_average': 'size_avg',
                                        'map_to_semantic': 'mts',
                                        'interval_validate': 'int_val',
                                        'resize_size': 'sz',
                                        'n_max_per_class': 'n_per',
                                        'semantic_subset': 'sem_set',
                                        'val_on_train': 'vot',
                                        'matching': 'match',
                                        'filter_images_by_semantic_subset': 'filter_sem',
                                        'set_extras_to_void': 'void',
                                        'momentum': 'm',
                                        'n_instances_per_class': 'n_inst_cls'}

BAD_CHAR_REPLACEMENTS = {' ': '', ',': '-', "['": '', "']": ''}


def check_clean_work_tree(exit_on_error=False, interactive=True):
    child = subprocess.Popen(['git', 'diff', '--name-only', '--exit-code'], stdout=subprocess.PIPE)
    stdout = child.communicate()[0]
    exit_code = child.returncode
    if exit_code != 0:
        'Your working directory tree isn\'t clean ({}).'.format(stdout.decode('utf-8'))
        override = False
        if interactive:
            override = 'y' == input('Your working directory tree isn\'t clean ({}).  '
                                    'Please commit or stash your changes. If you\'d like to run anyway,\n type \'y\''
                                    ''.format(stdout.decode()))
        if exit_on_error or interactive and not override:
            raise Exception('Exiting.  Please commit or stash your changes.')
    return exit_code, stdout


def create_config_copy(config_dict, config_key_replacements=CONFIG_KEY_REPLACEMENTS_FOR_FILENAME):
    cfg_print = config_dict.copy()
    for key, replacement_key in config_key_replacements.items():
        if key in cfg_print:
            cfg_print[replacement_key] = cfg_print.pop(key)
    return cfg_print


def create_config_from_default(config_args, default_config):
    cfg = default_config
    cfg.update(config_args)
    return cfg


def load_config(config_path):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_latest_logdir_with_checkpoint(log_parent_directory):
    checkpoint_paths = glob(os.path.join(log_parent_directory, '*/checkpoint.pth.tar'))
    checkpoint_paths.sort(key=os.path.getmtime)
    return os.path.dirname(checkpoint_paths[-1])


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
    hash = hash.decode("utf-8")
    return hash


def get_log_dir(model_name, config_id, cfg, parent_directory=None):
    bad_char_replacements = BAD_CHAR_REPLACEMENTS
    # load config
    name = 'MODEL-%s_CFG-%03d' % (model_name, config_id)
    for k, v in cfg.items():
        v = str(v)
        if '/' in v:
            continue
        name += '_%s-%s' % (k.upper(), v)
        for key, val in bad_char_replacements.items():
            name = name.replace(key, val)
    now = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))
    name += '_VCS-{}'.format(git_hash().replace("'", ""))
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
        torchfcn.models.FCN8sInstanceNotAtOnce,
        torchfcn.models.FCN8sAtOnce,
        torchfcn.models.FCN8sInstanceAtOnce
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
            import ipdb;
            ipdb.set_trace()
            raise ValueError('Unexpected module: %s' % str(m))
