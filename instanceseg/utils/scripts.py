import argparse
import os
import subprocess
from os import path as osp

import numpy as np
import torch
import torch.utils.data

import instanceseg.factory
import instanceseg.factory.data
import instanceseg.factory.models
import instanceseg.factory.optimizer
import instanceseg.factory.samplers
import instanceseg.factory.trainers
import instanceseg.utils
import instanceseg.utils.configs
import instanceseg.utils.logs
import scripts.configurations
from instanceseg.datasets import dataset_registry
from instanceseg.models import model_utils
from instanceseg.utils.configs import get_cfgs
from instanceseg.utils.misc import TermColors
from scripts.configurations.sampler_cfg import sampler_cfgs
from scripts.train_instances_filtered import here

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
    assert args.dataset is not None, ValueError('dataset argument must not be None.  Run with --help for more details.')
    cfg_default = dataset_registry.REGISTRY[args.dataset].default_config
    cfg_override_parser = instanceseg.utils.configs.get_cfg_override_parser(cfg_default)

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
                                      lambda old_val: convert_comma_separated_string_to_list(old_val, str),
                                      error_if_attr_doesnt_exist=False)
    replace_attr_with_function_of_val(override_cfg_args, 'img_size',
                                      lambda old_val: convert_comma_separated_string_to_list(old_val, int),
                                      error_if_attr_doesnt_exist=False)
    replace_attr_with_function_of_val(override_cfg_args, 'resize_size',
                                      lambda old_val: convert_comma_separated_string_to_list(old_val, int),
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


def convert_comma_separated_string_to_list(string, conversion_type=None):
    if conversion_type is None:
        conversion_type = str
    if string is None or string == '':
        return string
    else:
        elements = [s.strip() for s in string.split(',')]
        return [conversion_type(element) for element in elements]


def setup(dataset_type, cfg, out_dir, sampler_cfg, gpu=0, resume=None, semantic_init=None):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    set_random_seeds()
    cuda = torch.cuda.is_available()

    print('Getting dataloaders...')
    dataloaders = instanceseg.factory.data.get_dataloaders(cfg, dataset_type, cuda, sampler_cfg)
    print('Done getting dataloaders')
    # reduce dataloaders to semantic subset before running / generating problem config:
    n_instances_per_class = cfg['n_instances_per_class']
    problem_config = instanceseg.factory.models.get_problem_config(dataloaders['val'].dataset.semantic_class_names,
                                                                   n_instances_per_class,
                                                                   map_to_semantic=cfg['map_to_semantic'])
    if resume:
        checkpoint = torch.load(resume)
    else:
        checkpoint = None

    # 2. model
    model, start_epoch, start_iteration = instanceseg.factory.models.get_model(cfg, problem_config, resume,
                                                                               semantic_init, cuda)
    # Run a few checks
    problem_config_semantic_classes = set([problem_config.semantic_class_names[si]
                                           for si in problem_config.semantic_instance_class_list])
    dataset_semantic_classes = set(dataloaders['train'].dataset.semantic_class_names)
    assert problem_config_semantic_classes == dataset_semantic_classes, \
        'Model covers these semantic classes: {}.\n ' \
        'Dataset covers these semantic classes: {}.'.format(problem_config_semantic_classes, dataset_semantic_classes)
    print('Number of output channels in model: {}'.format(model.n_output_channels))
    print('Number of training, validation, train_for_val images: {}, {}, {}'.format(
        len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['train_for_val'] or 0)))

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    checkpoint_for_optim = checkpoint if (checkpoint is not None and not cfg['reset_optim']) else None
    optim = instanceseg.factory.optimizer.get_optimizer(cfg, model, checkpoint_for_optim)
    if cfg['freeze_vgg']:
        for module_name, module in model.named_children():
            if module_name in model_utils.VGG_CHILDREN_NAMES:
                assert all([p.requires_grad is False for p in module.parameters()])
        print('All modules were correctly frozen: '.format({}).format(model_utils.VGG_CHILDREN_NAMES))
    if not cfg['map_to_semantic']:
        cfg['activation_layers_to_export'] = tuple([x for x in cfg[
            'activation_layers_to_export'] if x is not 'conv1x1_instance_to_semantic'])
    trainer = instanceseg.factory.trainers.get_trainer(cfg, cuda, model, optim, dataloaders, problem_config, out_dir)
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    return trainer


def configure(dataset_name, config_idx, sampler_name, cfg_override_args=None):
    cfg, cfg_to_print = get_cfgs(dataset_name=dataset_name, config_idx=config_idx, cfg_override_args=cfg_override_args)
    cfg['sampler'] = sampler_name
    assert cfg['dataset'] == dataset_name, 'Debug Error: cfg[\'dataset\']: {}, args.dataset: {}'.format(cfg['dataset'],
                                                                                                        dataset_name)
    if cfg['dataset_instance_cap'] == 'match_model':
        cfg['dataset_instance_cap'] = cfg['n_instances_per_class']
    sampler_cfg = scripts.configurations.sampler_cfg.get_sampler_cfg(sampler_name)
    out_dir = instanceseg.utils.logs.get_log_dir(osp.basename(__file__).replace('.py', ''), config_idx,
                                                 cfg_to_print,
                                                 parent_directory=os.path.join(here, 'logs', dataset_name))
    instanceseg.utils.configs.save_config(out_dir, cfg)
    print(instanceseg.utils.misc.color_text('logdir: {}'.format(out_dir), instanceseg.utils.misc.TermColors.OKGREEN))
    return cfg, out_dir, sampler_cfg
