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
import scripts.configurations
from instanceseg.models import model_utils
from instanceseg.utils.configs import get_cfgs
from instanceseg.utils.misc import TermColors
from scripts.configurations.sampler_cfg_registry import sampler_cfgs

from local_pyutils import get_log_dir

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'

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
                TermColors.WARNING + 'Your working directory tree isn\'t clean:\n ' +
                TermColors.ENDC +
                TermColors.FAIL + '{}'.format(stdout.decode()) + TermColors.ENDC +
                'Please commit or stash your changes. If you\'d like to run anyway,\n enter \'y\': '
                '' + TermColors.ENDC)
        if exit_on_error or interactive and not override:
            raise Exception(
                TermColors.FAIL + 'Exiting.  Please commit or stash your changes.' +
                TermColors.ENDC)
    return exit_code, stdout


def setup_train(dataset_type, cfg, out_dir, sampler_cfg, gpu=(0,), checkpoint_path=None, semantic_init=None,
                splits=('train', 'val', 'train_for_val')):
    checkpoint, cuda, dataloaders, model, problem_config, start_epoch, start_iteration = \
        setup_common(dataset_type, cfg, gpu, checkpoint_path, sampler_cfg, semantic_init, splits=splits)
    print('Number of training, validation, train_for_val images: {}, {}, {}'.format(
        len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['train_for_val'] or 0)))

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    checkpoint_for_optim = checkpoint if (
            checkpoint is not None and not cfg['reset_optim']) else None
    optim = instanceseg.factory.optimizer.get_optimizer(cfg, model, checkpoint_for_optim)
    scheduler = instanceseg.factory.optimizer.get_scheduler(optim, cfg['lr_scheduler']) \
        if cfg['lr_scheduler'] is not None else None
    if cfg['freeze_vgg']:
        for module_name, module in model.named_children():
            if module_name in model_utils.VGG_CHILDREN_NAMES:
                assert all([p.requires_grad is False for p in module.parameters()])
        print(
            'All modules were correctly frozen: '.format({}).format(model_utils.VGG_CHILDREN_NAMES))
    if not cfg['map_to_semantic']:
        cfg['activation_layers_to_export'] = tuple([x for x in cfg[
            'activation_layers_to_export'] if x is not 'conv1x1_instance_to_semantic'])

    trainer = instanceseg.factory.trainers.get_trainer(cfg, cuda, model, dataloaders, problem_config, out_dir, optim,
                                                       scheduler=scheduler)
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    return trainer


def setup_test(dataset_type, cfg, out_dir, sampler_cfg, model_checkpoint_path, gpu=(0,), splits=('test',)):
    checkpoint, cuda, dataloaders, model, problem_config, start_epoch, start_iteration = \
        setup_common(dataset_type, cfg, gpu, model_checkpoint_path, sampler_cfg, semantic_init=None, splits=splits)
    print('Number of test minibatches: {}'.format(len(dataloaders[s]) for s in splits))

    evaluator = instanceseg.factory.trainers.get_evaluator(cfg, cuda, model, dataloaders, problem_config, out_dir)
    evaluator.epoch = start_epoch
    evaluator.iteration = start_iteration
    return evaluator


def setup_common(dataset_type, cfg, gpu, model_checkpoint_path, sampler_cfg, semantic_init,
                 splits=('train', 'val', 'train_for_val')):
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(['{}'.format(g) for g in gpu])
    print('CUDA_VISIBLE_DEVICES: ', os.environ['CUDA_VISIBLE_DEVICES'])
    set_random_seeds()
    cuda = torch.cuda.is_available()
    print('Using {} devices'.format(torch.cuda.device_count()))
    print('Getting dataloaders...')
    dataloaders = instanceseg.factory.data.get_dataloaders(cfg, dataset_type, cuda, sampler_cfg=sampler_cfg,
                                                           splits=splits)
    print('Done getting dataloaders')
    # reduce dataloaders to semantic subset before running / generating problem config:
    n_instances_per_class = cfg['n_instances_per_class']
    if len(splits) > 1:
        assert all(
            set(dataloaders[s].dataset.semantic_class_names) == set(dataloaders[splits[0]].dataset.semantic_class_names)
            for s in splits[1:])
    # labels_table = dataloaders[splits[0]].dataset.labels_table
    # NOTE(allie): Should double-check labels_table
    labels_table = dataloaders[splits[0]].dataset.labels_table
    n_instances_by_semantic_id = [1 if not l.isthing else n_instances_per_class for l in labels_table]
    problem_config = instanceseg.factory.models.get_problem_config_from_labels_table(
        labels_table, n_instances_by_semantic_id, map_to_semantic=cfg['map_to_semantic'])
    if model_checkpoint_path:
        checkpoint = torch.load(model_checkpoint_path)
    else:
        checkpoint = None
    # 2. model
    model, start_epoch, start_iteration = instanceseg.factory.models.get_model(cfg, problem_config,
                                                                               model_checkpoint_path,
                                                                               semantic_init, cuda)

    # Run a few checks
    problem_config_semantic_classes = set([problem_config.semantic_class_names[si]
                                           for si in problem_config.semantic_instance_class_list])
    dataset_semantic_classes = set(dataloaders[splits[0]].dataset.semantic_class_names)
    assert problem_config_semantic_classes == dataset_semantic_classes, \
        'Model covers these semantic classes: {}.\n ' \
        'Dataset covers these semantic classes: {}.'.format(problem_config_semantic_classes,
                                                            dataset_semantic_classes)
    print('Number of output channels in model: {}'.format(model.module.n_output_channels
                                                          if isinstance(model, torch.nn.DataParallel)
                                                          else model.n_output_channels))

    return checkpoint, cuda, dataloaders, model, problem_config, start_epoch, start_iteration


def configure(dataset_name, config_idx, sampler_name, script_py_file='unknownscript.py',
              cfg_override_args=None, parent_script_directory='scripts'):
    script_basename = osp.basename(script_py_file).replace('.py', '')
    parent_directory = os.path.join(PROJECT_ROOT, parent_script_directory, 'logs', dataset_name)
    cfg, cfg_to_print = get_cfgs(dataset_name=dataset_name, config_idx=config_idx,
                                 cfg_override_args=cfg_override_args)
    if sampler_name is not None or 'sampler' not in cfg:
        cfg['sampler'] = sampler_name
    else:
        sampler_name = cfg['sampler']
    assert cfg['dataset'] == dataset_name, 'Debug Error: cfg[\'dataset\']: {}, ' \
                                           'args.dataset: {}'.format(cfg['dataset'], dataset_name)
    if cfg['dataset_instance_cap'] == 'match_model':
        cfg['dataset_instance_cap'] = cfg['n_instances_per_class']
    sampler_cfg = scripts.configurations.sampler_cfg_registry.get_sampler_cfg_set(sampler_name)
    out_dir = get_log_dir(os.path.join(parent_directory, script_basename), cfg_to_print)
    instanceseg.utils.configs.save_config(out_dir, cfg)
    print(instanceseg.utils.misc.color_text('logdir: {}'.format(out_dir),
                                            instanceseg.utils.misc.TermColors.OKGREEN))
    return cfg, out_dir, sampler_cfg
