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
from torchfcn.datasets import pink_blobs

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)


# filename starts to exceed max; creating abbreviations so we can keep the config in the log
# directory name.
CONFIG_KEY_REPLACEMENTS_FOR_FILENAME = {'max_iteration': 'itr',
                                        'weight_decay': 'decay',
                                        'n_training_imgs': 'n_train',
                                        'n_validation_imgs': 'n_val',
                                        'recompute_optimal_loss': 'recompute_loss',
                                        'size_average': 'size_avg'}

default_configuration = dict(
    max_iteration=10000,
    lr=1.0e-5,
    weight_decay=5e-6,
    interval_validate=100,
    n_max_per_class=3,
    n_training_imgs=1000,
    n_validation_imgs=50,
    batch_size=1,
    recompute_optimal_loss=False,
    size_average=True,
    val_on_train=True)

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(),
    1: dict(
        batch_size=10),
    2: dict(
        n_training_imgs=10),
    3: dict(
        n_training_imgs=100),
    4: dict(
        recompute_optimal_loss=True,
        size_average=True
    ),
    5: dict(
        recompute_optimal_loss=True,
        size_average=False
    ),
    6: dict(
        recompute_optimal_loss=False,
        size_average=False
    ),
    7: dict(
        batch_size=2),
}

here = osp.dirname(osp.abspath(__file__))
logger = local_pyutils.get_logger()


def create_config_copy(config_dict, config_key_replacements=CONFIG_KEY_REPLACEMENTS_FOR_FILENAME):
    cfg_print = config_dict.copy()
    for key, replacement_key in config_key_replacements.items():
        cfg_print[replacement_key] = cfg_print.pop(key)
    return cfg_print


def main():
    matching = True
    assert_val_not_in_train = False

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=0,
                        choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = default_configuration
    cfg.update(configurations[args.config])
    out = get_log_dir(osp.basename(__file__).replace(
        '.py', ''), args.config, create_config_copy(cfg), parent_directory=osp.dirname(osp.abspath(
        __file__)))

    logger.info('logdir: {}'.format(out))
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_dataset = pink_blobs.BlobExampleGenerator(
        transform=True, n_max_per_class=cfg['n_max_per_class'], max_index=cfg['n_training_imgs']
                                                                          - 1)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['batch_size'],
                                               shuffle=True)
    if cfg['val_on_train']:
        train_dataset_for_val = train_dataset.copy(modified_length=10)
        train_loader_for_val = torch.utils.data.DataLoader(train_dataset_for_val, batch_size=1,
                                                           shuffle=False)
    else:
        train_loader_for_val = None
    val_dataset = pink_blobs.BlobExampleGenerator(transform=True,
                                                  n_max_per_class=cfg['n_max_per_class'],
                                                  max_index=cfg['n_validation_imgs'] - 1)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False)

    if assert_val_not_in_train:
        logger.info('Checking whether validation images appear in the training set.')
        dataset_utils.assert_validation_images_arent_in_training_set(train_loader, val_loader)
        logger.info('Confirmed validation and training set are disjoint.')

    # Make sure we can load an image
    [img, lbl] = train_loader.dataset[0]

    # 2. model
    # n_max_per_class > 1 and map_to_semantic=False: Basically produces extra channels that
    # should be '0'.  Not good if copying weights over from a pretrained semantic segmenter,
    # but fine otherwise.
    model = torchfcn.models.FCN8sInstance(
        n_semantic_classes_with_background=len(train_loader.dataset.class_names),
        n_max_per_class=cfg['n_max_per_class'],
        map_to_semantic=False)

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
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
        tensorboard_writer=writer,
        matching_loss=matching,
        recompute_loss_at_optimal_permutation=cfg['recompute_optimal_loss'],
        size_average=cfg.get('size_average'),
        train_loader_for_val=train_loader_for_val
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
