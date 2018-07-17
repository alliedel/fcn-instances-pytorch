import datetime
import os
import shlex
import subprocess
from collections import OrderedDict
from glob import glob
from os import path as osp

import numpy as np
import pytz
import torch
import yaml

from scripts.configurations.sampler_cfg import sampler_cfgs
from torchfcn.models import model_utils
from torchfcn.utils.optimizer import get_optimizer
from torchfcn.utils.samplers import get_sampler_cfg
from torchfcn.utils.scripts import here, MY_TIMEZONE, BAD_CHAR_REPLACEMENTS
from torchfcn.utils.misc import TermColors, color_text
from torchfcn.utils.trainers import get_trainer
from torchfcn.utils.models import get_model, get_problem_config
from torchfcn.utils.configs import load_config_from_logdir
from torchfcn.utils.data import get_dataloaders


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

    problem_config = get_problem_config(dataloaders['val'].dataset.semantic_class_names, n_instances_per_class,
                                        map_to_semantic=cfg['map_to_semantic'])

    checkpoint_file = resume

    # 2. model
    model, start_epoch, start_iteration = get_model(cfg, problem_config, checkpoint_file, semantic_init, cuda)

    print('Number of output channels in model: {}'.format(model.n_output_channels))
    print('Number of training, validation, train_for_val images: {}, {}, {}'.format(
        len(dataloaders['train']), len(dataloaders['val']), len(dataloaders['train_for_val'] or 0)))

    # 3. optimizer
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