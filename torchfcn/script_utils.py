#!/usr/bin/env python

import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess
from collections import OrderedDict
from glob import glob

import numpy as np
import pytz
import torch
import torch.utils.data
import yaml

from scripts.configurations import synthetic_cfg, voc_cfg
from scripts.configurations.sampler_cfg import sampler_cfgs
from torchfcn import instance_utils
from torchfcn.utils.configs import get_parameters, get_cfg_override_parser

here = osp.dirname(osp.abspath(__file__))
MY_TIMEZONE = 'America/New_York'

CONFIG_KEY_REPLACEMENTS_FOR_FILENAME = {'max_iteration': 'itr',
                                        'weight_decay': 'decay',
                                        'n_training_imgs': 'n_train',
                                        'n_validation_imgs': 'n_val',
                                        'recompute_optimal_loss': 'recomp',
                                        'size_average': 'sa',
                                        'map_to_semantic': 'mts',
                                        'interval_validate': 'val',
                                        'resize_size': 'sz',
                                        'n_max_per_class': 'n_per',
                                        'semantic_subset': 'sset',
                                        'val_on_train': 'VOT',
                                        'matching': 'ma',
                                        'set_extras_to_void': 'void',
                                        'momentum': 'mo',
                                        'n_instances_per_class': 'nper',
                                        'semantic_only_labels': 'sem_ls',
                                        'initialize_from_semantic': 'init_sem',
                                        'bottleneck_channel_capacity': 'bcc',
                                        'single_instance': '1inst',
                                        'score_multiplier': 'sm',
                                        'weight_by_instance': 'wt',
                                        'optim': 'o',
                                        'augment_semantic': 'augsem',
                                        'add_conv8': 'conv8',
                                        }

BAD_CHAR_REPLACEMENTS = {' ': '', ',': '-', "['": '', "']": ''}

CFG_ORDER = {}

DEBUG_ASSERTS = True


class TermColors:
    """
    https://stackoverflow.com/questions/287871/print-in-terminal-with-colors
    """
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def set_random_seeds(np_seed=1337, torch_seed=1337, torch_cuda_seed=1337):
    if np_seed is not None:
        np.random.seed(np_seed)
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    if torch_cuda_seed is not None:
        torch.cuda.manual_seed(torch_cuda_seed)


def str_or_int(val):
    try:
        return int(val)
    except ValueError:
        return val


def parse_args():
    # Get initial parser
    parser = get_parser(
        voc_default=voc_cfg.get_default_config(),
        voc_configs=voc_cfg.configurations,
        synthetic_default=synthetic_cfg.get_default_config(),
        synthetic_configs=synthetic_cfg.configurations,
    )

    args, argv = parser.parse_known_args()

    # Config override parser
    cfg_default = {'synthetic': synthetic_cfg.get_default_config(),
                   'voc': voc_cfg.get_default_config()}[args.dataset]
    cfg_override_parser = get_cfg_override_parser(cfg_default)

    bad_args = [arg for arg in argv[::2] if arg.replace('-', '') not in cfg_default.keys()]
    assert len(bad_args) == 0, cfg_override_parser.error('bad_args: {}'.format(bad_args))

    # Parse with list of options
    override_cfg_args, leftovers = cfg_override_parser.parse_known_args(argv)
    assert len(leftovers) == 0, ValueError('args not recognized: {}'.format(leftovers))
    # apparently this is failing, so I'm going to have to screen this on my own:

    # Remove options from namespace that weren't defined
    unused_keys = [k for k in list(override_cfg_args.__dict__.keys()) if '--' + k not in argv and '-' + k not in argv]
    for k in unused_keys:
        delattr(override_cfg_args, k)

    # Fix a few values
    replace_attr_with_function_of_val(override_cfg_args, 'clip', lambda old_val: old_val if old_val > 0 else None,
                                      error_if_attr_doesnt_exist=False)
    replace_attr_with_function_of_val(override_cfg_args, 'semantic_subset',
                                      lambda old_val: old_val if (old_val is None or old_val == '') else
                                      [s.strip() for s in old_val.split(',')],
                                      error_if_attr_doesnt_exist=False)

    return args, override_cfg_args


def replace_attr_with_function_of_val(namespace, attr, replacement_function, error_if_attr_doesnt_exist=True):
    if attr in namespace.__dict__.keys():
        setattr(namespace, attr, replacement_function(getattr(namespace, attr)))
    elif error_if_attr_doesnt_exist:
        raise Exception('attr {} does not exist in namespace'.format(attr))


def get_parser(voc_default, voc_configs, synthetic_default, synthetic_configs):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help='dataset: voc, synthetic', dest='dataset')
    dataset_parsers = {
        'voc': subparsers.add_parser('voc', help='VOC dataset options',
                                     epilog='\n\nOverride options:\n' + '\n'.join(
                                         ['--{}: {}'.format(k, v) for k, v in voc_default.items()]),
                                     formatter_class=argparse.RawTextHelpFormatter),
        'synthetic': subparsers.add_parser('synthetic', help='synthetic dataset options')
    }
    for dataset_name, subparser in dataset_parsers.items():
        cfg_choices = list({'synthetic': synthetic_configs,
                            'voc': voc_configs}[dataset_name].keys())
        subparser.add_argument('-c', '--config', type=str_or_int, default=0, choices=cfg_choices)
        subparser.add_argument('-g', '--gpu', type=int, required=True)
        subparser.add_argument('--resume', help='Checkpoint path')
        subparser.add_argument('--semantic-init', help='Checkpoint path of semantic model (e.g. - '
                                                       '\'~/data/models/pytorch/semantic_synthetic.pth\'', default=None)
        subparser.add_argument('--single-image-index', type=int, help='Image index to use for train/validation set',
                               default=None)
        subparser.add_argument('--sampler', type=str, choices=sampler_cfgs.keys(), default='default',
                               help='Sampler for dataset')
    return parser


def get_sampler_cfg(sampler_arg):
    sampler_cfg = sampler_cfgs[sampler_arg]
    if sampler_cfg['train_for_val'] is None:
        sampler_cfg['train_for_val'] = sampler_cfgs['default']['train_for_val']
    return sampler_cfg


def prune_defaults_from_dict(default_dict, update_dict):
    non_defaults = update_dict.copy()
    keys_to_pop = []
    for key in update_dict.keys():
        if key in default_dict and update_dict[key] == default_dict[key]:
            keys_to_pop.append(key)
    for key in keys_to_pop:
        non_defaults.pop(key)
    return non_defaults


def color_text(text, color):
    """
    color can either be a string, like 'OKGREEN', or the value itself, like TermColors.OKGREEN
    """
    color_keys = TermColors.__dict__.keys()
    color_vals = [getattr(TermColors, k) for k in color_keys]
    if color in color_keys:
        color = getattr(TermColors, color)
    elif color in color_vals:
        pass
    else:
        raise Exception('color not recognized: {}\nChoose from: {}, {}'.format(color, color_keys, color_vals))
    return color + text + TermColors.ENDC


def check_clean_work_tree(exit_on_error=False, interactive=True):
    child = subprocess.Popen(['git', 'diff', '--name-only', '--exit-code'], stdout=subprocess.PIPE)
    stdout = child.communicate()[0]
    exit_code = child.returncode
    if exit_code != 0:
        override = False
        if interactive:
            override = 'y' == input(
                TermColors.WARNING + 'Your working directory tree isn\'t clean:\n ' + TermColors.ENDC +
                TermColors.FAIL + '{}'.format(stdout.decode()) + TermColors.ENDC +
                'Please commit or stash your changes. If you\'d like to run anyway,\n enter \'y\': '
                '' + TermColors.ENDC)
        if exit_on_error or interactive and not override:
            raise Exception(TermColors.FAIL + 'Exiting.  Please commit or stash your changes.' + TermColors.ENDC)
    return exit_code, stdout


def create_config_copy(config_dict, config_key_replacements=CONFIG_KEY_REPLACEMENTS_FOR_FILENAME,
                       reverse_replacements=False):
    if reverse_replacements:
        config_key_replacements = {v: k for k, v in config_key_replacements.items()}
    cfg_print = config_dict.copy()
    for key, replacement_key in config_key_replacements.items():
        if key == 'semantic_subset' or key == config_key_replacements['semantic_subset']:
            if 'semantic_subset' in config_dict.keys() and config_dict['semantic_subset'] is not None:
                cfg_print['semantic_subset'] = '_'.join([cls.strip() for cls in config_dict['semantic_subset']])
        if key in cfg_print:
            cfg_print[replacement_key] = cfg_print.pop(key)

    return cfg_print


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
    hash_ = subprocess.check_output(shlex.split(cmd)).strip()
    hash_ = hash_.decode("utf-8")
    return hash_


def get_log_dir(model_name, config_id=None, cfg=None, parent_directory='logs'):
    bad_char_replacements = BAD_CHAR_REPLACEMENTS
    # load config
    now = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))
    name = 'TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    name += '_VCS-{}'.format(git_hash().replace("'", ""))
    name += '_MODEL-%s' % model_name
    if config_id is not None:
        if isinstance(config_id, int):
            name += '_CFG-%03d' % config_id
        else:
            name += '_CFG-{}'.format(config_id)
    if cfg is not None:
        for k, v in cfg.items():
            v = str(v)
            if '/' in v:
                continue
            if isinstance(v, list):
                import ipdb; ipdb.set_trace()
                v = '_'.join(v)
            name += '_%s-%s' % (k.upper(), v)
            for key, val in bad_char_replacements.items():
                name = name.replace(key, val)
    # create out
    if parent_directory is None:
        parent_directory = here
    log_dir = osp.join(parent_directory, name)
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        if isinstance(cfg, OrderedDict):
            yaml.safe_dump(dict(cfg), f, default_flow_style=False)
        else:
            yaml.safe_dump(cfg, f, default_flow_style=False)
    return log_dir


def save_config(log_dir, cfg):
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def get_optimizer(cfg, model, checkpoint_file=None):
    if cfg['optim'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optim'] == 'sgd':
        optim = torch.optim.SGD(
            [
                {'params': filter(lambda p: False if p is None else p.requires_grad,
                                  get_parameters(model, bias=False))},
                {'params': filter(lambda p: False if p is None else p.requires_grad,
                                  get_parameters(model, bias=True)),
                 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
    else:
        raise Exception('optimizer {} not recognized.'.format(cfg['optim']))
    if checkpoint_file:
        checkpoint = torch.load(checkpoint_file)
        optim.load_state_dict(checkpoint['optim_state_dict'])
    return optim


# val_dataset.class_names
def get_problem_config(class_names, n_instances_per_class: int, map_to_semantic=False):
    # 0. Problem setup (instance segmentation definition)
    class_names = class_names
    n_semantic_classes = len(class_names)
    n_instances_by_semantic_id = [1] + [n_instances_per_class for _ in range(1, n_semantic_classes)]
    problem_config = instance_utils.InstanceProblemConfig(n_instances_by_semantic_id=n_instances_by_semantic_id,
                                                          map_to_semantic=map_to_semantic)
    problem_config.set_class_names(class_names)
    return problem_config


def pairwise_or(list1, list2):
    return [a or b for a, b in zip(list1, list2)]


