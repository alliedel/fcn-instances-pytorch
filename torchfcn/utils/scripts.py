import argparse
import subprocess
from os import path as osp

import numpy as np
import torch
import torch.utils.data

from scripts.configurations.sampler_cfg import sampler_cfgs
from torchfcn.datasets import dataset_registry
import torchfcn.utils.configs
from torchfcn.utils.misc import TermColors

here = osp.dirname(osp.abspath(__file__))
MY_TIMEZONE = 'America/New_York'
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


def get_parser():
    # voc_default=voc_cfg.get_default_config(),
    # voc_configs=voc_cfg.configurations,
    # synthetic_default=synthetic_cfg.get_default_config(),
    # synthetic_configs=synthetic_cfg.configurations,

    # voc_default, voc_configs, synthetic_default, synthetic_configs
    parser = argparse.ArgumentParser()
    dataset_names = dataset_registry.REGISTRY.keys()
    subparsers = parser.add_subparsers(help='dataset: {}'.format(dataset_names), dest='dataset')
    dataset_parsers = {
        dataset_name:
            subparsers.add_parser(dataset_name, help='{} dataset options'.format(dataset_name),
                                  epilog='\n\nOverride options:\n' + '\n'.join(
                                      ['--{}: {}'.format(k, v)
                                       for k, v in dataset_registry.REGISTRY[dataset_name].default_config.items()]),
                                  formatter_class=argparse.RawTextHelpFormatter)
        for dataset_name in dataset_names
    }
    for dataset_name, subparser in dataset_parsers.items():
        cfg_choices = list(dataset_registry.REGISTRY[dataset_name].config_options.keys())
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


def parse_args():
    # Get initial parser
    parser = get_parser()

    args, argv = parser.parse_known_args()

    # Config override parser
    cfg_default = dataset_registry.REGISTRY[args.dataset].default_config
    cfg_override_parser = torchfcn.utils.configs.get_cfg_override_parser(cfg_default)

    bad_args = [arg for arg in argv[::2] if arg.replace('-', '') not in cfg_default.keys()]
    assert len(bad_args) == 0, cfg_override_parser.error('bad_args: {}'.format(bad_args))
    if args.sampler is not None:
        argv += ['--sampler', args.sampler]
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


def str_or_int(val):
    try:
        return int(val)
    except ValueError:
        return val
