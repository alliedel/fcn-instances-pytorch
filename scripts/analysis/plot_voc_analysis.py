#!/usr/bin/env python

import argparse
import numbers
import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm

import torchfcn
from examples.voc.script_utils import get_log_dir

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    1: dict(
        max_iteration=100000,
        lr=1.0e-10,
        momentum=0.99,
        weight_decay=0.0005,
        interval_validate=4000,
    )
}

here = osp.dirname(osp.abspath(__file__))


def main():
    n_max_per_class = 100
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

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    # 1. dataset
    semantic_subset = None  # ['background', 'person']
    root = osp.expanduser('~/data/datasets')
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    train_dataset = torchfcn.datasets.VOC2012ClassSeg(root, split='train', transform=True,
                                                      semantic_subset=semantic_subset,
                                                      n_max_per_class=n_max_per_class,
                                                      permute_instance_order=False,
                                                      set_extras_to_void=True,
                                                      return_semantic_instance_tuple=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False)

    # Make sure we can load an image
    try:
        [img, lbl] = train_loader.dataset[0]
    except:
        raise

    val_dataset = torchfcn.datasets.VOC2012ClassSeg(root, split='val', transform=True,
                                                    semantic_subset=semantic_subset,
                                                    n_max_per_class=n_max_per_class,
                                                    permute_instance_order=False,
                                                    set_extras_to_void=True,
                                                    return_semantic_instance_tuple=True)
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=1, shuffle=False, **kwargs)

    splits = ['train', 'val']
    split = splits[0]
    dataset_summary_file = '/tmp/{}_dataset_summary.npz'.format(split)
    key_file = '/tmp/{}_key.npz'.format(split)
    train_dataset_summary, train_key = load_dataset_summary(dataset_summary_file, key_file)
    plot_summary_histograms(train_dataset_summary)


def load_dataset_summary(summary_file, key_file):
    val_d = dict(np.load(summary_file).iteritems())
    key_d = dict(np.load(key_file).iteritems())
    return val_d, key_d


def plot_summary_histograms(train_dataset_summary, figure_dir='/tmp'):
    for i, (k, v) in tqdm.tqdm(enumerate(train_dataset_summary.iteritems()),
                               total=len(train_dataset_summary.keys()), ncols=80,
                               desc='Generating plots...', leave=True):
        if type(v) is not np.ndarray:
            print('Ignoring {} of type {}'.format(v, type(v)))
            continue
        if not isinstance(v[0], numbers.Number):
            print('Ignoring {} whose elements are of type {}'.format(v, type(v)))
            continue
        plt.figure(1)
        plt.clf()
        bins = np.linspace(v.min(), v.max(), 50)
        if len(v.shape) == 1:
            plt.hist(v, bins, alpha=0.8, label='{}: {}'.format(k, 'all'))
            plt.legend()
            plt.savefig('{}/{}.png'.format(figure_dir, k))
        else:
            if v.shape[1] > 1000:
                print('Ignoring {}, shape {}: would require {} plots'.format(k, v.shape,
                                                                             v.shape[1]))
                print(v.shape)
                continue
            for idx in tqdm.tqdm(range(v.shape[1]), total=v.shape[1], ncols=80,
                                 desc='Subplots for data {}'.format(k), leave=False):
                plt.clf()
                plt.hist(v[:, idx], bins, alpha=0.8, label='{}: {}'.format(k, idx))
                plt.legend()
                plt.savefig('{}/{}_{}.png'.format(figure_dir, k, idx))


if __name__ == '__main__':
    main()
