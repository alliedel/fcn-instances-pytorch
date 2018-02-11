#!/usr/bin/env python 

import argparse
import os
import os.path as osp

import torch

import torchfcn
from torchfcn.datasets import dataset_utils

from examples.voc.script_utils import get_log_dir
from examples.voc.script_utils import get_parameters

from tensorboardX import SummaryWriter
from torchfcn.datasets import pink_blobs

import local_pyutils

logger = local_pyutils.get_logger()


default_configuration = dict(
    max_iteration=10000,
    lr=1.0e-10,
    momentum=0.99,
    weight_decay=0.0005,
    interval_validate=100,
    n_max_per_class=3,
    n_training_imgs=100,
    n_validation_imgs=50,
    batch_size=1)

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(),
    1: dict(
        batch_size=10),
    2: dict(
        n_training_imgs=10),
    3: dict(
        n_training_imgs=1000,
    )
}

here = osp.dirname(osp.abspath(__file__))


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
        '.py', ''), args.config, cfg, parent_directory=osp.dirname(osp.abspath(__file__)))

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
    val_dataset = pink_blobs.BlobExampleGenerator(transform=True,
                                                  n_max_per_class=cfg['n_max_per_class'],
                                                  max_index=cfg['n_validation_imgs'] - 1)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False)
    logger.info('Checking whether validation images appear in the training set.')
    if assert_val_not_in_train:
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

    optim = torch.optim.SGD(
        [
            {'params': filter(lambda p: False if p is None else p.requires_grad, get_parameters(
                model, bias=False))},
            {'params': filter(lambda p: False if p is None else p.requires_grad, get_parameters(
                model, bias=True)),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])
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
        matching_loss=matching
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
