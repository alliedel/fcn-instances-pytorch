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

from numpy import floor

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=1000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=100,
    )
}

here = osp.dirname(osp.abspath(__file__))


def main():
    n_max_per_class = 3
    matching = True
    clrs = [pink_blobs.Defaults.clrs[0] for _ in range(4)]



    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('-c', '--config', type=int, default=1,
                        choices=configurations.keys())
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    cfg = configurations[args.config]
    out = get_log_dir(osp.basename(__file__).replace(
        '.py', ''), args.config, cfg, parent_directory=osp.dirname(osp.abspath(__file__)))

    print('logdir: {}'.format(out))
    resume = args.resume

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    root = osp.expanduser('~/data/datasets')

    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    velocity_multiplier = 4
    velocity_r_c = [[v * velocity_multiplier for v in vel]
                    for vel in pink_blobs.Defaults.velocity_r_c]

    train_dataset = pink_blobs.PinkBlobExampleGenerator(
        transform=True, n_max_per_class=n_max_per_class, velocity_r_c=velocity_r_c,
        clrs=clrs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
    offset_c, offset_r = [velocity_multiplier / 2, velocity_multiplier / 2]
    val_initial_cols = [c + offset_c for c in train_dataset.initial_cols]
    val_initial_rows = [r + offset_r for r in train_dataset.initial_rows]
    val_dataset = pink_blobs.PinkBlobExampleGenerator(transform=True,
                                                      n_max_per_class=n_max_per_class,
                                                      initial_cols=val_initial_cols,
                                                      initial_rows=val_initial_rows,
                                                      velocity_r_c=velocity_r_c, max_index=2)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False)

    dataset_utils.assert_validation_images_arent_in_training_set(train_loader, val_loader)

    # Make sure we can load an image
    [img, lbl] = train_loader.dataset[0]

    # 2. model
    # n_max_per_class > 1 and map_to_semantic=False: Basically produces extra channels that
    # should be '0'.  Not good if copying weights over from a pretrained semantic segmenter,
    # but fine otherwise.
    model = torchfcn.models.FCN8sInstance(
        n_semantic_classes_with_background=len(train_loader.dataset.class_names),
        n_max_per_class=n_max_per_class,
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
