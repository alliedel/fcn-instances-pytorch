import argparse
from collections import OrderedDict
from os import path as osp

import yaml

import torchfcn
from local_pyutils import str2bool


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
