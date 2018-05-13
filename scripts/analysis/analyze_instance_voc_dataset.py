#!/usr/bin/env python

import argparse
import os
import os.path as osp

import numpy as np
import torch
import tqdm

import torchfcn
from examples.voc.script_utils import get_log_dir
from torchfcn.datasets import dataset_utils
from torchfcn.datasets import voc

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

    val_dataset_summary, val_key = create_dataset_summary(val_dataset, n_max_per_class=22)
    np.savez('/tmp/val_dataset_summary.npz', **val_dataset_summary)
    np.savez('/tmp/val_key.npz', **val_key)
    train_dataset_summary, train_key = create_dataset_summary(train_dataset, n_max_per_class=39)
    np.savez('/tmp/train_dataset_summary.npz', **train_dataset_summary)
    np.savez('/tmp/train_key.npz', **train_key)


def get_n_max_per_class_from_dataset(dataset):
    n_max_per_class = 0
    max_sem_idx = 0
    for idx, (data, (sem_target, inst_target)) in tqdm.tqdm(
            enumerate(dataset), total=len(dataset),
            desc='Finding max instances in dataset...', ncols=80,
            leave=False):
        n_max_per_class = max(n_max_per_class, torch.max(inst_target)) + 1
        max_sem_idx = max(max_sem_idx, torch.max(sem_target))
    assert max_sem_idx == len(voc.ALL_VOC_CLASS_NAMES) - 1, 'The dataset contains fewer / more ' \
                                                            'than the semantic classes I thought'
    print('\n Max instances: {}'.format(n_max_per_class))
    return n_max_per_class


def create_dataset_summary(dataset, n_max_per_class=None):
    # Get # max instances
    if n_max_per_class is None:
        n_max_per_class = get_n_max_per_class_from_dataset(dataset)
    dataset.update_n_max_per_class(n_max_per_class)

    # Get some stats
    semantic_classes = voc.ALL_VOC_CLASS_NAMES
    semantic_instance_mapping = dataset.get_instance_to_semantic_mapping()
    per_instance_semantic_names = dataset.get_instance_semantic_labels()
    n_imgs = len(dataset)
    # Initialize scalars
    number_of_pixels = np.empty((n_imgs,))
    number_of_void_pixels = np.empty((n_imgs,))
    number_of_total_instances = np.empty((n_imgs,))
    number_of_pixels_per_semantic = np.empty((n_imgs, len(semantic_classes)))
    number_of_pixels_per_instance_num = np.empty((n_imgs, n_max_per_class))
    number_of_pixels_per_semantic_instance = np.empty((n_imgs, len(per_instance_semantic_names)))
    number_of_instances_per_semantic_class = np.empty((n_imgs, len(per_instance_semantic_names)))
    image_names = []
    # Get counts for each image
    for idx, (data, (sem_target, inst_target)) in tqdm.tqdm(
            enumerate(dataset), total=len(dataset),
            desc='Analyzing dataset...', ncols=80,
            leave=False):
        image_names.append(osp.basename(dataset.files[dataset.split][idx]['img']))
        number_of_pixels[idx] = torch.numel(sem_target)
        number_of_pixels_per_semantic[idx, :] = [torch.sum(sem_target == sem_cls)
                                                 for sem_cls, _ in enumerate(semantic_classes)]
        number_of_pixels_per_instance_num[idx, :] = [torch.sum(inst_target == inst_num)
                                                     for inst_num in range(n_max_per_class)]
        number_of_instances_per_semantic_class[idx, :] = \
            [torch.max(inst_target[sem_target == sem_cls]) + 1
             if (torch.sum(sem_target == sem_cls) > 0) else 0
             for sem_cls, _ in enumerate(semantic_classes)]
        number_of_total_instances[idx] = np.sum(number_of_instances_per_semantic_class[idx, :])
        number_of_void_pixels[idx] = torch.sum(sem_target == -1)
        # The function below overwrites the inst_target, so gotta be careful!
        full_instance_target = dataset_utils.combine_semantic_and_instance_labels(sem_target,
                                                                                  inst_target,
                                                                                  n_max_per_class)
        number_of_pixels_per_semantic_instance[idx, :] = [torch.sum(full_instance_target ==
                                                                    sem_inst_cls)
                                                          for sem_inst_cls in
                                                          range(len(per_instance_semantic_names))]

    summary = {'image_names': image_names,
               'number_of_pixels': number_of_pixels,
               'number_of_pixels_per_semantic': number_of_pixels_per_semantic,
               'number_of_pixels_per_instance_num': number_of_pixels_per_instance_num,
               'number_of_pixels_per_semantic_instance': number_of_pixels_per_semantic_instance,
               'number_of_instances_per_semantic_class': number_of_instances_per_semantic_class,
               'number_of_void_pixels': number_of_void_pixels,
               'number_of_total_instances': number_of_total_instances
               }
    key = {
        'semantic_names': semantic_classes,
        'per_instance_semantic_names': per_instance_semantic_names,
        'semantic_instance_mapping': semantic_instance_mapping
    }

    return summary, key


if __name__ == '__main__':
    main()
