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
    train_loader, val_loader, train_loader_for_val = get_dataset_loaders(cuda, cfg)
    semantic_train_ids = [ci for ci, sem_id in enumerate(train_loader.dataset.train_id_list)
                          if train_loader.dataset.train_id_assignments['semantic']]
    # TODO(allie): handle the case where we don't have consecutive semantic vals.
    number_per_instance_class = cfg['n_max_per_class']
    n_instances_by_semantic_id = [
        number_per_instance_class if train_loader.dataset.train_id_assignments['instance'][sem_id]
        else 1 for sem_id in semantic_train_ids]

    problem_config = instance_utils.InstanceProblemConfig(
        semantic_vals=semantic_train_ids,
        n_instances_by_semantic_id=n_instances_by_semantic_id,
        void_value=train_loader.dataset.void_value)

    # 2. model
    model = torchfcn.models.FCN8sInstance(
        problem_config.n_classes, map_to_semantic=cfg['map_to_semantic'],
        instance_to_semantic_mapping_matrix=problem_config.instance_to_semantic_mapping_matrix)

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

    # 3. optimizer
    optim = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])

    if resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    writer = SummaryWriter(log_dir=out)
    trainer = torchfcn.Trainer(
        cuda=cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out,
        max_iter=cfg.get('max_iteration'),
        interval_validate=cfg.get('interval_validate', len(train_loader)),
        tensorboard_writer=writer,
        matching_loss=cfg.get('matching'),
        recompute_loss_at_optimal_permutation=cfg.get('recompute_optimal_loss'),
        size_average=cfg.get('size_average'),
        train_loader_for_val=train_loader_for_val
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    import ipdb; ipdb.set_trace()
    trainer.train()


def get_dataset_loaders(cuda, cfg):
    root = osp.expanduser('~/data/cityscapes')
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    dataset_kwargs = {'resize': cfg['resize_size'] is not None,
                      'resize_size': cfg['resize_size']}
    train_dataset = torchfcn.datasets.CityscapesMappedToInstances(
        root, split='train', **dataset_kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True,
                                               batch_size=cfg['batch_size'], **loader_kwargs)
    if cfg['val_on_train']:
        train_dataset_for_val = train_dataset.copy(modified_length=10)
        train_loader_for_val = torch.utils.data.DataLoader(train_dataset_for_val, batch_size=1,
                                                           shuffle=False, **loader_kwargs)
    else:
        train_loader_for_val = None
    val_dataset_full = torchfcn.datasets.CityscapesMappedToInstances(root, split='val',
                                                                     **dataset_kwargs)
    val_dataset = val_dataset_full.copy(modified_length=50)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False, **loader_kwargs)
    if assert_val_not_in_train:
        logger.info('Checking whether validation images appear in the training set.')
        dataset_utils.assert_validation_images_arent_in_training_set(train_loader, val_loader)
        logger.info('Confirmed validation and training set are disjoint.')
    # Make sure we can load an image
    try:
        img, (sem_lbl, inst_lbl) = train_loader.dataset[0]
    except:
        print('Can\'t load an image/label pair.')
        import ipdb;
        ipdb.set_trace()
        raise
    # Test out the image transformations to make sure they look reasonable
    # torch.save(train_loader.dataset.untransform_img(img), 'untransformed_img.pth')
    # torch.save(train_loader.dataset.untransform_lbl(lbl), 'untransformed_lbl.pth')

    return train_loader, val_loader, train_loader_for_val


if __name__ == '__main__':
    main()
