#!/usr/bin/env python

import datetime
import os
import os.path as osp
import shlex
import subprocess

import pytz
import yaml

import torch
from glob import glob
from torchfcn.datasets.voc import VOC_ROOT
from torchfcn import instance_utils
from tensorboardX import SummaryWriter
import torchfcn
import torchfcn.datasets.voc
import torchfcn.datasets.synthetic
import numpy as np

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
                                        'filter_images_by_semantic_subset': 'f_sem',
                                        'set_extras_to_void': 'void',
                                        'momentum': 'mo',
                                        'n_instances_per_class': 'nper',
                                        'semantic_only_labels': 'sem_ls',
                                        'initialize_from_semantic': 'init_sem',
                                        'bottleneck_channel_capacity': 'bcc',
                                        'single_instance': '1inst',
                                        'score_multiplier': 'sm',
                                        'weight_by_instance': 'wt',
                                        'optim': 'o'
                                        }

BAD_CHAR_REPLACEMENTS = {' ': '', ',': '-', "['": '', "']": ''}


class bcolors:
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


def prune_defaults_from_dict(default_dict, update_dict):
    keys = update_dict.keys()
    keys_to_pop = []
    for key in keys:
        if update_dict[key] == default_dict[key]:
            keys_to_pop.append(key)
    for key in keys_to_pop:
        update_dict.pop(key)


def check_clean_work_tree(exit_on_error=False, interactive=True):
    child = subprocess.Popen(['git', 'diff', '--name-only', '--exit-code'], stdout=subprocess.PIPE)
    stdout = child.communicate()[0]
    exit_code = child.returncode
    if exit_code != 0:
        override = False
        if interactive:
            override = 'y' == input(bcolors.WARNING + 'Your working directory tree isn\'t clean:\n ' + bcolors.ENDC +
                                    bcolors.FAIL + '{}'.format(stdout.decode()) + bcolors.ENDC +
                                    'Please commit or stash your changes. If you\'d like to run anyway,\n enter \'y\': '
                                    '' + bcolors.ENDC)
        if exit_on_error or interactive and not override:
            raise Exception(bcolors.FAIL + 'Exiting.  Please commit or stash your changes.' + bcolors.ENDC)
    return exit_code, stdout


def create_config_copy(config_dict, config_key_replacements=CONFIG_KEY_REPLACEMENTS_FOR_FILENAME,
                       reverse_replacements=False):
    if reverse_replacements:
        config_key_replacements = {v: k for k, v in config_key_replacements.items()}
    cfg_print = config_dict.copy()
    for key, replacement_key in config_key_replacements.items():
        if key == 'semantic_subset':
            if config_dict['semantic_subset'] is not None:
                cfg_print['semantic_subset'] = ''.join([cls[0] for cls in config_dict['semantic_subset']])
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


def get_log_dir(model_name, config_id=None, cfg=None, parent_directory='logs'):
    bad_char_replacements = BAD_CHAR_REPLACEMENTS
    # load config
    now = datetime.datetime.now(pytz.timezone(MY_TIMEZONE))
    name = 'TIME-%s' % now.strftime('%Y%m%d-%H%M%S')
    name += '_VCS-{}'.format(git_hash().replace("'", ""))
    name += '_MODEL-%s' % model_name
    if config_id is not None:
        name += '_CFG-%03d' % config_id
    if cfg is not None:
        for k, v in cfg.items():
            v = str(v)
            if '/' in v:
                continue
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


def get_trainer(cfg, cuda, model, optim, dataloaders, problem_config, out_dir):
    writer = SummaryWriter(log_dir=out_dir)
    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=dataloaders['train'],
        val_loader=dataloaders['val'],
        train_loader_for_val=dataloaders['train_for_val'],
        instance_problem=problem_config,
        out=out_dir,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(dataloaders['train'])),
        tensorboard_writer=writer,
        matching_loss=cfg['matching'],
        loader_semantic_lbl_only=cfg['semantic_only_labels'],
        size_average=cfg['size_average']
    )
    return trainer


def get_optimizer(cfg, model, checkpoint=None):
    if cfg['optim'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optim'] == 'sgd':
        optim = torch.optim.SGD(
            [
                {'params': get_parameters(model, bias=False)},
                {'params': get_parameters(model, bias=True),
                 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
    else:
        raise Exception('optimizer {} not recognized.'.format(cfg['optim']))
    if checkpoint:
        optim.load_state_dict(checkpoint['optim_state_dict'])
    return optim


# val_dataset.class_names
def get_problem_config(class_names, n_instances_per_class):
    # 0. Problem setup (instance segmentation definition)
    class_names = class_names
    n_semantic_classes = len(class_names)
    n_instances_by_semantic_id = [1] + [n_instances_per_class for sem_cls in range(1, n_semantic_classes)]
    problem_config = instance_utils.InstanceProblemConfig(n_instances_by_semantic_id=n_instances_by_semantic_id)
    problem_config.set_class_names(class_names)
    return problem_config


def get_model(cfg, problem_config, checkpoint, semantic_init, cuda):
    model = torchfcn.models.FCN8sInstanceAtOnce(
        semantic_instance_class_list=problem_config.semantic_instance_class_list,
        map_to_semantic=False, include_instance_channel0=False,
        bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'], score_multiplier_init=cfg['score_multiplier'])
    print('Number of classes in model: {}'.format(model.n_classes))
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        start_epoch, start_iteration = 0, 0
        if cfg['initialize_from_semantic']:
            semantic_init_path = os.path.expanduser(semantic_init)
            if not os.path.exists(semantic_init_path):
                raise ValueError('I could not find the path {}.  Did you set the path using the semantic-init '
                                 'flag?'.format(semantic_init_path))
            semantic_model = torchfcn.models.FCN8sInstanceAtOnce(
                semantic_instance_class_list=[1 for _ in range(problem_config.n_semantic_classes)],
                map_to_semantic=False, include_instance_channel0=False)
            print('Copying params from preinitialized semantic model')
            checkpoint = torch.load(semantic_init_path)
            semantic_model.load_state_dict(checkpoint['model_state_dict'])
            model.copy_params_from_semantic_equivalent_of_me(semantic_model)
        else:
            print('Copying params from vgg16')
            vgg16 = torchfcn.models.VGG16(pretrained=True)
            model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()
    return model, start_epoch, start_iteration


def get_synthetic_datasets(cfg):
    synthetic_generator_n_instances_per_semantic_id = 2
    dataset_kwargs = dict(transform=True, n_max_per_class=synthetic_generator_n_instances_per_semantic_id,
                          map_to_single_instance_problem=cfg['single_instance'])
    train_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    val_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs)
    return train_dataset, val_dataset


def get_voc_datasets(cfg, voc_root):
    dataset_kwargs = dict(transform=True, semantic_only_labels=cfg['semantic_only_labels'],
                          set_extras_to_void=cfg['set_extras_to_void'], semantic_subset=cfg['semantic_subset'],
                          map_to_single_instance_problem=cfg['single_instance'])
    semantic_subset_as_str = cfg['semantic_subset']
    if semantic_subset_as_str is not None:
        semantic_subset_as_str = '_'.join(cfg['semantic_subset'])
    else:
        semantic_subset_as_str = cfg['semantic_subset']
    instance_counts_cfg_str = '_semantic_subset-{}'.format(semantic_subset_as_str)
    instance_counts_file = osp.expanduser('~/data/datasets/VOC/instance_counts{}.npy'.format(instance_counts_cfg_str))
    if os.path.exists(instance_counts_file):
        print('Loading precomputed instance counts from {}'.format(instance_counts_file))
        instance_precomputed = True
        instance_counts = np.load(instance_counts_file)
        if len(instance_counts.shape) == 0:
            raise Exception('instance counts file contained empty array. Delete it: {}'.format(instance_counts_file))
    else:
        print('No precomputed instance counts (checked in {})'.format(instance_counts_file))
        instance_precomputed = False
        instance_counts = None
    train_dataset_kwargs = dict(instance_counts_precomputed=instance_counts)
    train_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(voc_root, split='train', **dataset_kwargs,
                                                          **train_dataset_kwargs)
    if not instance_precomputed:
        try:
            assert train_dataset.instance_counts is not None
            np.save(instance_counts_file, train_dataset.instance_counts)
        except:
            import ipdb; ipdb.set_trace()  # to save from rage-quitting after having just computed the instance counts
            raise
    val_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(voc_root, split='seg11valid', **dataset_kwargs)
    return train_dataset, val_dataset


def get_dataloaders(cfg, dataset, cuda):
    # 1. dataset
    if dataset == 'synthetic':
        train_dataset, val_dataset = get_synthetic_datasets(cfg)
    elif dataset == 'voc':
        train_dataset, val_dataset = get_voc_datasets(cfg, VOC_ROOT)
    else:
        raise ValueError
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kwargs)
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset.copy(modified_length=3), batch_size=1,
                                                       shuffle=False, **loader_kwargs)
    try:
        img, (sl, il) = train_dataset[0]
    except:
        import ipdb; ipdb.set_trace()
        raise Exception('Cannot load an image from your dataset')

    return {
        'train': train_loader,
        'val': val_loader,
        'train_for_val': train_loader_for_val,
    }
