#!/usr/bin/env python

import argparse
import os
import os.path as osp
import numpy as np


import torch
from torchfcn.datasets import dataset_generator_registry, dataset_registry, dataset_statistics
from torchfcn.analysis import visualization_utils
import display_pyutils
import matplotlib.pyplot as plt


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


def get_instance_counts(dataset_type):
    default_train_dataset, default_val_dataset, transformer_tag = \
        dataset_generator_registry.get_default_datasets_for_instance_counts(dataset_type)
    default_datasets = {
        'train': default_train_dataset,
        'val': default_val_dataset,
    }

    # Compute statistics
    split = 'train'
    instance_count_file = os.path.join(dataset_registry.REGISTRY[dataset_type].dataset_path,
                                       '{}_instance_counts_{}.npy'.format(split, transformer_tag))
    if not os.path.isfile(instance_count_file):
        print('Generating file {}'.format(instance_count_file))
        stats = dataset_statistics.InstanceDatasetStatistics(default_datasets[split])
        stats.compute_statistics(filename_to_write_instance_counts=instance_count_file)
    else:
        print('Reading from instance counts file {}'.format(instance_count_file))
        stats = dataset_statistics.InstanceDatasetStatistics(
            default_datasets[split], existing_instance_count_file=instance_count_file)
    train_stats = stats

    split = 'val'
    instance_count_file = os.path.join(dataset_registry.REGISTRY[dataset_type].dataset_path,
                                       '{}_instance_counts_{}.npy'.format(split, transformer_tag))
    if not os.path.isfile(instance_count_file):
        print('Generating file {}'.format(instance_count_file))
        stats = dataset_statistics.InstanceDatasetStatistics(default_datasets[split])
        stats.compute_statistics(filename_to_write_instance_counts=instance_count_file)
    else:
        print('Reading from instance counts file {}'.format(instance_count_file))
        stats = dataset_statistics.InstanceDatasetStatistics(
            default_datasets[split], existing_instance_count_file=instance_count_file)
    val_stats = stats
    return train_stats, val_stats, default_train_dataset, default_val_dataset


def write_stats(train_stats, val_stats, default_train_dataset, default_val_dataset):
    # Write statistics
    # Data to plot
    for split in ['train', 'val']:
        dataset = default_train_dataset if split == 'train' else default_val_dataset
        stats = train_stats if split == 'train' else val_stats

        instance_counts = stats.instance_counts.numpy()
        semantic_class_names = dataset.semantic_class_names if split == 'train' else \
            dataset.semantic_class_names

        assert instance_counts.shape[0] == len(dataset)
        if instance_counts.shape[1] != len(semantic_class_names):
            import ipdb; ipdb.set_trace()
            raise Exception

        instance_counts_per_class = instance_counts.sum(axis=0)
        image_counts_per_class = (instance_counts > 0).sum(axis=0)

        plt.figure(1); plt.clf()
        metric_name = split + '_' + 'instance_counts_per_class'
        metric_values = instance_counts_per_class
        labels = [s + '={}'.format(v) for s, v in zip(semantic_class_names, metric_values)]
        pie_chart(metric_values, labels)
        plt.title(metric_name)
        display_pyutils.save_fig_to_workspace(metric_name + '.png')

        plt.figure(2); plt.clf()
        metric_name = split + '_' + 'image_counts_per_class'
        metric_values = image_counts_per_class
        labels = [s + '={}'.format(v) for s, v in zip(semantic_class_names, metric_values)]
        pie_chart(metric_values, labels)
        plt.title(metric_name)
        display_pyutils.save_fig_to_workspace(metric_name + '.png')


def pie_chart(values, labels, autopct='%1.1f%%'):
    explode = [0.1 * int(val != max(values)) for i, val in enumerate(values)]  # explode largest slice
    # Plot
    colors = [display_pyutils.GOOD_COLORS[np.mod(i, len(display_pyutils.GOOD_COLORS))] for i in range(len(values))]
    plt.pie(values, explode=explode, labels=labels, colors=colors,
            autopct=autopct, shadow=True, startangle=140)
    plt.axis('equal')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    train_stats, val_stats, default_train_dataset, default_val_dataset = get_instance_counts('cityscapes')
    write_stats(train_stats, val_stats, default_train_dataset, default_val_dataset)


if __name__ == '__main__':
    main()
