import argparse
from collections import OrderedDict
from os import path as osp

import yaml

import instanceseg
import instanceseg.utils
from instanceseg.datasets import dataset_registry
from graveyard.models import attention_old
from instanceseg.utils.misc import str2bool
from . import misc

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
                                        'dataset_instance_cap': 'datacap',
                                        'export_activations': 'exp_act',
                                        'write_instance_metrics': 'instmet',
                                        'loss_type': 'loss',
                                        'ordering': 'order',
                                        'reset_optim': 'ropt'
                                        }

CONFIG_VAL_REPLACEMENTS_FOR_FILENAME = {
    True: '0',
    False: '1'
}


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


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        nn.Softmax,
        instanceseg.models.FCN8sInstance,
        instanceseg.models.resnet_instance.ResNet50Instance
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
        elif isinstance(m, attention_old.Self_Attn):
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


def create_config_copy(config_dict, config_key_replacements='default',
                       reverse_replacements=False, config_val_replacements='default'):
    if config_val_replacements is 'default':
        config_val_replacements = CONFIG_VAL_REPLACEMENTS_FOR_FILENAME
    if config_key_replacements is 'default':
        config_key_replacements = CONFIG_KEY_REPLACEMENTS_FOR_FILENAME
    if reverse_replacements:
        config_key_replacements = {v: k for k, v in config_key_replacements.items()}
        config_val_replacements = {v: k for k, v in config_val_replacements.items()}
    cfg_print = config_dict.copy()
    for key, replacement_key in config_key_replacements.items():
        if key == 'semantic_subset' or key == config_key_replacements['semantic_subset']:
            if 'semantic_subset' in config_dict.keys() and config_dict['semantic_subset'] is not None:
                cfg_print['semantic_subset'] = '_'.join([cls.strip() for cls in config_dict['semantic_subset'] if cls
                                                         is not 'background'])
        if key in cfg_print:
            cfg_print[replacement_key] = cfg_print.pop(key)

    for key, old_val in cfg_print.items():
        if old_val in config_val_replacements.keys():
            cfg_print[key] = config_val_replacements[old_val]

    return cfg_print


def save_config(log_dir, cfg):
    with open(osp.join(log_dir, 'config.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f, default_flow_style=False)


def get_cfgs(dataset_name, config_idx, cfg_override_args=None):
    cfg_default = dataset_registry.REGISTRY[dataset_name].default_config
    cfg_options = dataset_registry.REGISTRY[dataset_name].config_options
    cfg = create_config_from_default(cfg_options[config_idx], cfg_default)
    non_default_options = prune_defaults_from_dict(cfg_default, cfg)
    if cfg_override_args is not None:
        try:
            cfg_override_as_dict = cfg_override_args.__dict__
        except AttributeError:
            cfg_override_as_dict = cfg_override_args
        for key, override_val in cfg_override_as_dict.items():
            old_val = cfg.pop(key)
            if override_val != old_val:
                print(misc.color_text(
                    'Overriding value for {}: {} --> {}'.format(key, old_val, override_val),
                    misc.TermColors.WARNING))
            cfg[key] = override_val
            non_default_options[key] = override_val

    if cfg['semantic_subset'] is not None and 'background' not in cfg['semantic_subset']:
        print(UserWarning('background was not in the list of semantic classes.  I added it.  '
                          'If you truly don\'t want it, deal with this code.'))
        cfg['semantic_subset'].append('background')

    print(misc.color_text('non-default cfg values: {}'.format(non_default_options),
                          misc.TermColors.OKBLUE))
    cfg_to_print = non_default_options
    cfg_to_print = create_config_copy(cfg_to_print)
    cfg_to_print = make_ordered_cfg(cfg_to_print)

    return cfg, cfg_to_print


def prune_defaults_from_dict(default_dict, update_dict):
    non_defaults = update_dict.copy()
    keys_to_pop = []
    for key in update_dict.keys():
        if key in default_dict and update_dict[key] == default_dict[key]:
            keys_to_pop.append(key)
    for key in keys_to_pop:
        non_defaults.pop(key)
    return non_defaults
