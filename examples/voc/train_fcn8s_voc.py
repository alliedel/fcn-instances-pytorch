#!/usr/bin/env python

import argparse
import os
import os.path as osp

import local_pyutils
import numpy as np
import torch
from tensorboardX import SummaryWriter

import torchfcn
from examples.script_utils import get_log_dir
from torchfcn.datasets import dataset_utils
from torchfcn import instance_utils
import config

# filename starts to exceed max; creating abbreviations so we can keep the config in the log
# directory name.

here = osp.dirname(osp.abspath(__file__))
logger = local_pyutils.get_logger()

# Script args (don't affect performance)
assert_val_not_in_train = False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=0,
                        choices=config.configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gpu = args.gpu
    cfg = config.get_config(args.config)
    if cfg['no_inst']:
        assert cfg['n_max_per_class'] == 1, "no_inst implies n_max_per_class=1.  Please change " \
                                            "the value accordingly."

    out = get_log_dir(osp.basename(__file__).replace(
        '.py', ''), args.config, config.create_config_copy(cfg),
        parent_directory=osp.dirname(osp.abspath(__file__)))

    logger.info('logdir: {}'.format(out))
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    np.random.seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    dataset_kwargs = {
        'cutoff_instances_above': cfg['n_max_per_class'],
        'make_all_instance_classes_semantic': cfg['no_inst']}
    root = os.path.expanduser('~/data/datasets/')
    train_dataset = torchfcn.datasets.voc.VOCMappedToInstances(root, split='train',
                                                               **dataset_kwargs)
    val_dataset_full = torchfcn.datasets.voc.VOCMappedToInstances(root, split='val',
                                                                  **dataset_kwargs)
    val_dataset = val_dataset_full.copy(modified_length=50)
    train_loader, val_loader, train_loader_for_val = dataset_utils.get_dataset_loaders(
        cuda, cfg, train_dataset, val_dataset)
    
    # 2. model
    semantic_train_ids = [ci for ci, sem_id in enumerate(train_loader.dataset.train_id_list)
                          if train_loader.dataset.train_id_assignments['semantic'][ci]]
    number_per_instance_class = cfg['n_max_per_class']
    n_instances_by_semantic_id = [
        number_per_instance_class if train_loader.dataset.train_id_assignments['instance'][sem_id]
        else 1 for sem_id in semantic_train_ids]
    problem_config = instance_utils.InstanceProblemConfig(
        semantic_vals=semantic_train_ids,
        n_instances_by_semantic_id=n_instances_by_semantic_id,
        void_value=train_loader.dataset.void_value)
    model = torchfcn.models.FCN8sInstance(
        problem_config.n_classes, map_to_semantic=cfg['map_to_semantic'],
        semantic_instance_class_list=problem_config.semantic_instance_class_list)

    # 3. initialize/resume model
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # 4. optimizer
    if cfg['opt'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['opt'] == 'sgd':
        optim = torch.optim.SGD(
            [
                {'params': filter(lambda p: True if p is None else p.requires_grad, get_parameters(
                    model, bias=False))},
                {'params': filter(lambda p: True if p is None else p.requires_grad,
                                  get_parameters(model, bias=True)),
                 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
    else:
        raise Exception('opt={} not recognized'.format(cfg['opt']))
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    # 5. train
    writer = SummaryWriter(log_dir=out)
    trainer = torchfcn.Trainer(cuda=cuda, model=model, optimizer=optim, train_loader=train_loader,
                               val_loader=val_loader, out=out, max_iter=cfg.get('max_iteration'),
                               interval_validate=cfg.get('interval_validate', len(train_loader)),
                               tensorboard_writer=writer, matching_loss=cfg.get('matching'),
                               recompute_loss_at_optimal_permutation=cfg['recompute_optimal_loss'],
                               size_average=cfg.get('size_average'),
                               train_loader_for_val=train_loader_for_val)
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    logger.info('Starting training.')
    trainer.train()


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
        torchfcn.models.FCN8sInstance,
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
            import ipdb; ipdb.set_trace()
            raise ValueError('Unexpected module: %s' % str(m))


if __name__ == '__main__':
    main()
