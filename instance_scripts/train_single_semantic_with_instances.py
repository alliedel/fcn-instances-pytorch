#!/usr/bin/env python

import argparse
import os
import os.path as osp
import socket

import torch

import torchfcn

from local_pyutils import get_log_dir
from torchfcn.models.model_utils import get_parameters
from torchfcn import instance_trainer


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-14,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    )
}


here = osp.dirname(osp.abspath(__file__))


def run_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    log_dir = get_log_dir('logs/fcn8s-instance', cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    args.cuda = cuda

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
    return args, log_dir, cfg


def get_datasets(args):
    if socket.gethostname() == 'allieLaptop-Ubuntu':
        root = osp.expanduser('~/data/pascal/')
    elif socket.gethostname() == 'kalman':
        root = osp.expanduser('~/data/datasets/pascal/')
    else:
        raise Exception('Specify dataset root for hostname {}'.format(socket.gethostname()))
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2007ObjectSeg(
            root, split='train', transform=True),
        batch_size=1, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        torchfcn.datasets.VOC2007ObjectSeg(
            root, split='val', transform=True),
        batch_size=1, shuffle=False, **kwargs)
    return train_loader, val_loader


def main():
    # 0. Config/setup
    args, out_log_dir, cfg = run_setup()

    # 1. dataset
    train_loader, val_loader = get_datasets(args)

    # 2. model
    model = torchfcn.models.FCN8sInstance(n_semantic_classes=21)
    start_epoch = 0
    start_iteration = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        vgg16 = torchfcn.models.VGG16(pretrained=True)
        model.copy_params_from_vgg16(vgg16)
    if args.cuda:
        model = model.cuda()

    # 3. optimizer

    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': cfg['lr'] * 2, 'weight_decay': 0},
        ],
        lr=cfg['lr'],
        momentum=cfg['momentum'],
        weight_decay=cfg['weight_decay'])
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])

    trainer = instance_trainer.InstanceTrainer(
        cuda=args.cuda,
        model=model,
        optimizer=optim,
        train_loader=train_loader,
        val_loader=val_loader,
        out=out_log_dir,
        max_iter=cfg['max_iteration'],
        interval_validate=cfg.get('interval_validate', len(train_loader)),
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()


if __name__ == '__main__':
    main()
