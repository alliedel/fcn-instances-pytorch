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
from local_pyutils import TermColors
from tensorboardX import SummaryWriter

import torchfcn
import torchfcn.datasets.synthetic
import torchfcn.datasets.voc
from scripts.configurations import synthetic_cfg, voc_cfg
from scripts.configurations.sampler_cfg import sampler_cfgs
from scripts.train_instances_filtered import str2bool
from torchfcn import instance_utils
from torchfcn.datasets import dataset_statistics, samplers
from torchfcn.datasets.voc import VOC_ROOT
from torchfcn.models import model_utils
from torchfcn.models import attention

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
                                        'use_conv8': 'conv8',
                                        }

BAD_CHAR_REPLACEMENTS = {' ': '', ',': '-', "['": '', "']": ''}

CFG_ORDER = {}

DEBUG_ASSERTS = True


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


def make_ordered_cfg(cfg, start_arg_order=('dataset', 'sampler'), end_arg_order=()):
    """
    Returns an ordered copy of cfg, with start, <alphabetical>, and end ordering.
    """
    # Get order
    cfg_keys = list(cfg.keys())
    start_keys = [k for k in start_arg_order if k in cfg_keys]
    end_keys = [k for k in end_arg_order if k in cfg_keys]
    for k in start_keys + end_keys:
        cfg_keys.remove(k)
    middle_keys = [k for k in sorted(cfg_keys)]

    # Add keys in order
    cfg_ordered = OrderedDict()
    for k in start_keys + middle_keys + end_keys:
        cfg_ordered[k] = cfg[k]
    return cfg_ordered


def create_config_from_default(config_args, default_config):
    cfg = default_config.copy()
    cfg.update(config_args)
    return cfg


def load_config_from_logdir(logdir):
    cfg_file = osp.join(logdir, 'config.yaml')
    return load_config(cfg_file)


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


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        nn.Softmax,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
        torchfcn.models.FCN8sAtOnce,
        torchfcn.models.FCN8sInstance
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, attention.Self_Attn):
            if bias:
                yield m.query_conv.bias
                yield m.key_conv.bias
                yield m.value_conv.bias
            else:
                yield m.query_conv.weight
                yield m.key_conv.weight
                yield m.value_conv.weight
                yield m.gamma
        elif isinstance(m, modules_skipped):
            continue
        else:
            import ipdb
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
        size_average=cfg['size_average'],
        augment_input_with_semantic_masks=cfg['augment_semantic'],
        export_activations=cfg['export_activations'],
        activation_layers_to_export=cfg['activation_layers_to_export'],
        bool_compute_instance_metrics=cfg['write_instance_metrics'],
        generate_new_synthetic_data_each_epoch=(cfg['dataset'] == 'synthetic' and cfg['infinite_synthetic'])
    )
    return trainer


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


def get_model(cfg, problem_config, checkpoint_file, semantic_init, cuda):
    n_input_channels = 3 if not cfg['augment_semantic'] else 3 + problem_config.n_semantic_classes
    try:
        model = torchfcn.models.FCN8sInstance(
            semantic_instance_class_list=problem_config.model_semantic_instance_class_list,
            map_to_semantic=problem_config.map_to_semantic, include_instance_channel0=False,
            bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'], score_multiplier_init=cfg['score_multiplier'],
            n_input_channels=n_input_channels, clip=cfg['clip'], use_conv8=cfg['use_conv8'], use_attention_layer=cfg[
                'use_attn_layer'])
    except:
        print('Warning: deprecated.')
        model = torchfcn.models.FCN8sInstance(
            semantic_instance_class_list=problem_config.model_semantic_instance_class_list,
            map_to_semantic=problem_config.map_to_semantic, include_instance_channel0=False,
            bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'], score_multiplier_init=cfg['score_multiplier'],
            n_input_channels=n_input_channels, clip=cfg['clip'])
    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
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
            semantic_model = torchfcn.models.FCN8sInstance(
                semantic_instance_class_list=[1 for _ in range(problem_config.n_semantic_classes)],
                map_to_semantic=False, include_instance_channel0=False)
            print('Copying params from preinitialized semantic model')
            checkpoint_file = torch.load(semantic_init_path)
            semantic_model.load_state_dict(checkpoint_file['model_state_dict'])
            model.copy_params_from_semantic_equivalent_of_me(semantic_model)
        else:
            print('Copying params from vgg16')
            vgg16 = torchfcn.models.VGG16(pretrained=True)
            model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    if cfg['freeze_vgg']:
        model_utils.freeze_vgg_module_subset(model)
    return model, start_epoch, start_iteration


def get_synthetic_datasets(cfg, transform=True):
    dataset_kwargs = dict(transform=transform, n_max_per_class=cfg['synthetic_generator_n_instances_per_semantic_id'],
                          map_to_single_instance_problem=cfg['single_instance'], ordering=cfg['ordering'],
                          semantic_subset=cfg['semantic_subset'], one_dimension=cfg['one_dimension'])
    train_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs, n_images=cfg.pop(
        'n_images_train', None))
    val_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs, n_images=cfg.pop(
        'n_images_val', None))
    return train_dataset, val_dataset


def get_voc_datasets(cfg, voc_root, transform=True):
    dataset_kwargs = dict(transform=transform, semantic_only_labels=cfg['semantic_only_labels'],
                          set_extras_to_void=cfg['set_extras_to_void'],
                          map_to_single_instance_problem=cfg['single_instance'],
                          ordering=cfg['ordering'])
    train_dataset_kwargs = dict()
    train_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(voc_root, split='train', **dataset_kwargs,
                                                          **train_dataset_kwargs)
    val_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(voc_root, split='seg11valid', **dataset_kwargs)
    return train_dataset, val_dataset


def get_sampler(dataset_instance_stats, sequential, sem_cls=None, n_instances_range=None, n_images=None):
    valid_indices = [True for _ in range(len(dataset_instance_stats.dataset))]
    if n_instances_range is not None:
        valid_indices = pairwise_and(valid_indices,
                                     dataset_instance_stats.filter_images_by_n_instances(n_instances_range, sem_cls))
    elif sem_cls is not None:
        valid_indices = pairwise_and(valid_indices, dataset_instance_stats.filter_images_by_semantic_classes(sem_cls))
    if n_images is not None:
        if sum(valid_indices) < n_images:
            raise Exception('Too few images to sample {}.  Choose a smaller value for n_images in the sampler '
                            'config, or change your filtering requirements for the sampler.'.format(n_images))

        # Subsample n_images
        n_images_chosen = 0
        for idx in np.random.permutation(len(valid_indices)):
            if valid_indices[idx]:
                if n_images_chosen == n_images:
                    valid_indices[idx] = False
                else:
                    n_images_chosen += 1
        try:
            assert sum(valid_indices) == n_images
        except AssertionError:
            import ipdb
            ipdb.set_trace()
            raise
    sampler = samplers.sampler_factory(sequential=sequential, bool_index_subset=valid_indices)(
        dataset_instance_stats.dataset)

    return sampler


def get_configured_sampler(dataset_type, dataset, sequential, n_instances_range, n_images, sem_cls_filter,
                           instance_count_file):
    if n_instances_range is not None:
        if dataset_type != 'voc':
            raise NotImplementedError('Need an established place to save instance counts')
        instance_counts = torch.from_numpy(np.load(instance_count_file)) \
            if os.path.isfile(instance_count_file) else None
        stats = dataset_statistics.InstanceDatasetStatistics(dataset, instance_counts)
        if instance_counts is None:
            stats.compute_statistics()
            instance_counts = stats.instance_counts
            np.save(instance_count_file, instance_counts.numpy())
    else:
        stats = dataset_statistics.InstanceDatasetStatistics(dataset)

    my_sampler = get_sampler(stats, sequential=sequential, n_instances_range=n_instances_range, sem_cls=sem_cls_filter,
                             n_images=n_images)
    if n_images:
        assert len(my_sampler.indices) == n_images
    return my_sampler


def get_dataloaders(cfg, dataset_type, cuda, sampler_cfg=None):
    # 1. dataset
    if dataset_type == 'synthetic':
        train_dataset, val_dataset = get_synthetic_datasets(cfg)
    elif dataset_type == 'voc':
        train_dataset, val_dataset = get_voc_datasets(cfg, VOC_ROOT)
        if cfg['semantic_subset'] is not None:
            train_dataset.reduce_to_semantic_subset(cfg['semantic_subset'])
            val_dataset.reduce_to_semantic_subset(cfg['semantic_subset'])
        instance_cap = cfg['n_instances_per_class'] if cfg['dataset_instance_cap'] == 'match_model' else \
            cfg['dataset_instance_cap']
        if instance_cap is not None:
            train_dataset.set_instance_cap(instance_cap)
            val_dataset.set_instance_cap(instance_cap)
    else:
        raise ValueError('dataset_type={} not recognized'.format(dataset_type))

    # 2. samplers
    if sampler_cfg is None:
        train_sampler = samplers.sampler.RandomSampler(train_dataset)
        val_sampler = samplers.sampler.SequentialSampler(val_dataset)
        train_for_val_sampler = samplers.sampler.SequentialSampler(train_dataset)
    else:
        train_sampler_cfg = sampler_cfg['train']
        val_sampler_cfg = sampler_cfg['val']
        train_for_val_cfg = pop_without_del(sampler_cfg, 'train_for_val', None)
        sampler_cfg['train_for_val'] = train_for_val_cfg

        sem_cls_filter = pop_without_del(train_sampler_cfg, 'sem_cls_filter', None)
        if sem_cls_filter is not None:
            if isinstance(sem_cls_filter[0], str):
                try:
                    sem_cls_filter = [train_dataset.class_names.index(class_name) for class_name in sem_cls_filter]
                except:
                    sem_cls_filter = [int(np.where(train_dataset.class_names == class_name)[0][0])
                                      for class_name in sem_cls_filter]
        train_instance_count_file = os.path.join(VOC_ROOT, 'train_instance_counts.npy')
        train_sampler = get_configured_sampler(dataset_type, train_dataset, sequential=True,
                                               n_instances_range=pop_without_del(train_sampler_cfg,
                                                                                 'n_instances_range', None),
                                               n_images=pop_without_del(train_sampler_cfg, 'n_images', None),
                                               sem_cls_filter=sem_cls_filter,
                                               instance_count_file=train_instance_count_file)
        if isinstance(val_sampler_cfg, str) and val_sampler_cfg == 'copy_train':
            val_sampler = train_sampler.copy(sequential_override=True)
            val_dataset = train_dataset
        else:
            sem_cls_filter = pop_without_del(val_sampler_cfg, 'sem_cls_filter', None)
            if sem_cls_filter is not None:
                if isinstance(sem_cls_filter[0], str):
                    try:
                        sem_cls_filter = [val_dataset.class_names.index(class_name) for class_name in sem_cls_filter]
                    except:
                        sem_cls_filter = [int(np.where(val_dataset.class_names == class_name)[0][0])
                                          for class_name in sem_cls_filter]
            val_instance_count_file = os.path.join(VOC_ROOT, 'val_instance_counts.npy')
            val_sampler = get_configured_sampler(dataset_type, val_dataset, sequential=True,
                                                 n_instances_range=pop_without_del(val_sampler_cfg,
                                                                                   'n_instances_range', None),
                                                 n_images=pop_without_del(val_sampler_cfg, 'n_images', None),
                                                 sem_cls_filter=sem_cls_filter,
                                                 instance_count_file=val_instance_count_file)

        cut_n_images = pop_without_del(train_for_val_cfg, 'n_images', None) or len(train_dataset)
        train_for_val_sampler = train_sampler.copy(sequential_override=True,
                                                   cut_n_images=None if cut_n_images is None
                                                   else min(cut_n_images, len(train_sampler)))

    # Create dataloaders from datasets and samplers
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=val_sampler, **loader_kwargs)
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                       sampler=train_for_val_sampler, **loader_kwargs)

    if DEBUG_ASSERTS:
        try:
            i, [sl, il] = [d for i, d in enumerate(train_loader) if i == 0][0]
        except:
            raise

    return {
        'train': train_loader,
        'val': val_loader,
        'train_for_val': train_loader_for_val,
    }


def pairwise_and(list1, list2):
    return [a and b for a, b in zip(list1, list2)]


def pairwise_or(list1, list2):
    return [a or b for a, b in zip(list1, list2)]


def pop_without_del(dictionary, key, default):
    val = dictionary.pop(key, default)
    dictionary[key] = val
    return val


def load_everything_from_logdir(logdir, gpu=0, packed_as_dict=False):
    cfg = load_config_from_logdir(logdir)
    dataset_name = cfg['dataset']
    if dataset_name != os.path.basename(os.path.dirname(os.path.normpath(logdir))):
        cfg_dataset_name = dataset_name
        dataset_name = os.path.basename(os.path.dirname(os.path.normpath(logdir)))
        print(color_text('cfg[\'dataset\'] was set to '
                         '{} but I think based on the log directory it\'s actually '
                         '{}'.format(cfg_dataset_name, dataset_name), TermColors.WARNING))
    if dataset_name is None:
        print(color_text(
            'dataset not set in cfg -- this needs to be fixed for future experiments (supporting for '
            'legacy experiments).  Interpreting dataset name from folder name now...', TermColors.WARNING))
        dataset_name = os.path.basename(os.path.dirname(os.path.normpath(logdir)))
    model_pth = osp.join(logdir, 'model_best.pth.tar')
    out_dir = '/tmp'

    problem_config, model, trainer, optim, dataloaders = load_everything_from_cfg(cfg, gpu, dataset_name,
                                                                                  resume=model_pth, semantic_init=None,
                                                                                  out_dir=out_dir)
    if packed_as_dict:
        return dict(cfg=cfg, model_pth=model_pth, out_dir=out_dir, problem_config=problem_config, model=model,
                    trainer=trainer, optim=optim, dataloaders=dataloaders)
    else:
        return cfg, model_pth, out_dir, problem_config, model, trainer, optim, dataloaders


def load_everything_from_cfg(cfg: dict, gpu: int, dataset_name: str, resume: str, semantic_init, out_dir: str) -> tuple:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    np.random.seed(1234)
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
    print('Getting dataloaders...')
    sampler_cfg = get_sampler_cfg(cfg['sampler'])
    try:
        sampler_cfg['train_for_val']
    except:
        sampler_cfg['train_for_val'] = None
    if 'train_for_val' not in sampler_cfg.keys() or sampler_cfg['train_for_val'] is None:
        sampler_cfg['train_for_val'] = sampler_cfgs['default']['train_for_val']
    if 'n_images_train' in cfg.keys() and cfg['n_images_train'] is not None:
        sampler_cfg['train']['n_images'] = cfg['n_images_train']
    if 'n_images_val' in cfg.keys() and cfg['n_images_val'] is not None:
        sampler_cfg['val']['n_images'] = cfg['n_images_val']
    if 'n_images_train_for_val' in cfg.keys() and cfg['n_images_train_for_val'] is not None:
        sampler_cfg['train_for_val']['n_images'] = cfg['n_images_train_for_val']
    dataloaders = get_dataloaders(cfg, dataset_name, cuda, sampler_cfg)
    print('Done getting dataloaders')
    try:
        i, [sl, il] = [d for i, d in enumerate(dataloaders['train']) if i == 0][0]
    except:
        raise
    n_instances_per_class = 1 if cfg['single_instance'] else cfg['n_instances_per_class']
    assert n_instances_per_class is not None

    problem_config = get_problem_config(dataloaders['val'].dataset.class_names, n_instances_per_class,
                                        map_to_semantic=cfg['map_to_semantic'])

    checkpoint_file = resume

    # 2. model
    model, start_epoch, start_iteration = get_model(cfg, problem_config, checkpoint_file, semantic_init, cuda)

    print('Number of output channels in model: {}'.format(model.n_output_channels))
    print('Number of training, validation, train_for_val images: {}, {}, {}'.format(
        len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['train_for_val'] or 0)))

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    optim = get_optimizer(cfg, model, checkpoint_file)

    if cfg['freeze_vgg']:
        for module_name, module in model.named_children():
            if module_name in model_utils.VGG_CHILDREN_NAMES:
                assert all([p for p in module.parameters()])
        print('All modules were correctly frozen: '.format({}).format(model_utils.VGG_CHILDREN_NAMES))
    trainer = get_trainer(cfg, cuda, model, optim, dataloaders, problem_config, out_dir)
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration

    return problem_config, model, trainer, optim, dataloaders


def get_cfg_override_parser(cfg_default):
    cfg_override_parser = argparse.ArgumentParser()

    for arg, default_val in cfg_default.items():
        if default_val is not None:
            arg_type = str2bool if isinstance(default_val, bool) else type(default_val)
            cfg_override_parser.add_argument('--' + arg, type=arg_type, default=default_val,
                                             help='cfg override (only recommended for one-off experiments '
                                                  '- set cfg in file instead)')
        else:
            cfg_override_parser.add_argument('--' + arg, default=default_val,
                                             help='cfg override (only recommended for one-off experiments '
                                                  '- set cfg in file instead)')
    return cfg_override_parser


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
