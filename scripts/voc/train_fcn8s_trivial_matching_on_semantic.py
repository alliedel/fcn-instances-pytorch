#!/usr/bin/env python

import argparse
import os
import os.path as osp

import numpy as np
import torch
from tensorboardX import SummaryWriter

import torchfcn
import torchfcn.datasets.voc
from torchfcn import instance_utils
from torchfcn import script_utils

here = osp.dirname(osp.abspath(__file__))


def main():
    script_utils.check_clean_work_tree()
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=0, choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()
    gpu = args.gpu
    config_idx = args.config

    cfg = script_utils.create_config_from_default(configurations[config_idx], default_config)

    out = script_utils.get_log_dir(osp.basename(__file__).replace(
        '.py', ''), config_idx, script_utils.create_config_copy(cfg),
        parent_directory=osp.dirname(osp.abspath(__file__)))

    print('logdir: {}'.format(out))
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 0. dataset
    root = osp.expanduser('~/data/datasets')

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
    train_dataset_kwargs = dict(weight_by_instance=cfg['weight_by_instance'],
                                instance_counts_precomputed=instance_counts)
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(root, split='train', **dataset_kwargs, **train_dataset_kwargs)
    if not instance_precomputed:
        try:
            assert train_dataset.instance_counts is not None
            np.save(instance_counts_file, train_dataset.instance_counts)
        except:
            import ipdb; ipdb.set_trace()  # to save from rage-quitting after having just computed the instance counts
            raise
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, **kwargs)
    val_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(root, split='seg11valid', **dataset_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, **kwargs)
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset.copy(modified_length=15), batch_size=1,
                                                       shuffle=False, **kwargs)

    # 1. Problem setup (instance segmentation definition)

    class_names = val_dataset.class_names
    n_semantic_classes = len(class_names)
    if cfg['single_instance'] and cfg['n_instances_per_class'] is not None and cfg['n_instances_per_class'] != 1:
        raise ValueError('n_instances_per_class should be 1 (or None) when running semantic loss')
    if cfg['n_instances_per_class'] is None:
        n_instances_per_class = 1 if cfg['single_instance'] else None
    else:
        n_instances_per_class = cfg['n_instances_per_class']
    n_instances_by_semantic_id = [1] + [n_instances_per_class for sem_cls in range(1, n_semantic_classes)]
    problem_config = instance_utils.InstanceProblemConfig(n_instances_by_semantic_id=n_instances_by_semantic_id)
    problem_config.set_class_names(val_dataset.class_names)

    # 2. model

    model = torchfcn.models.FCN8sInstanceAtOnce(
        semantic_instance_class_list=problem_config.semantic_instance_class_list, map_to_semantic=False)
    print('Number of classes in model: {}'.format(model.n_classes))
    start_epoch = 0
    start_iteration = 0
    if resume:
        checkpoint = torch.load(resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        print('Copying params from vgg16')
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    # 3. optimizer

    # 3. optimizer
    # TODO(allie): something is wrong with adam... fix it.
    if cfg['optim'] == 'adam':
        optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    elif cfg['optim'] == 'sgd':
        optim = torch.optim.SGD(
            [
                {'params': script_utils.get_parameters(model, bias=False)},
                {'params': script_utils.get_parameters(model, bias=True),
                 #            {'params': filter(lambda p: False if p is None else p.requires_grad, get_parameters(
                 #                model, bias=False))},
                 #            {'params': filter(lambda p: False if p is None else p.requires_grad, get_parameters(
                 #                model, bias=True)),

                 'lr': cfg['lr'] * 2, 'weight_decay': 0},
            ],
            lr=cfg['lr'],
            momentum=cfg['momentum'],
            weight_decay=cfg['weight_decay'])
    else:
        raise Exception('optimizer {} not recognized.'.format(cfg['optim']))
    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    writer = SummaryWriter(log_dir=out)
    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        instance_problem=problem_config,
        out=out,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
        tensorboard_writer=writer,
        matching_loss=cfg['matching'],
        loader_semantic_lbl_only=cfg['semantic_only_labels'],
        train_loader_for_val=train_loader_for_val,
        export_analytics_every=1000,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
