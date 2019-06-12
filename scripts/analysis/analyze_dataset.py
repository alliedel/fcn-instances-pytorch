#!/usr/bin/env python

import argparse
import os
import os.path as osp

import instanceseg.utils.display as display_pyutils
import matplotlib.pyplot as plt
import numpy as np

from instanceseg.datasets import dataset_generator_registry, dataset_registry, dataset_statistics

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
    instance_counts = {}
    # Compute statistics
    for split in default_datasets.keys():
        default_dataset = default_datasets[split]
        semantic_class_names = default_dataset.semantic_class_names
        instance_count_file = \
            dataset_registry.REGISTRY[dataset_type].get_instance_count_filename(split,
                                                                                transformer_tag)
        instance_count_cache = dataset_statistics.NumberofInstancesPerSemanticClass(
            range(len(semantic_class_names)), cache_file=instance_count_file)
        instance_counts[split] = instance_count_cache.stat_tensor
    train_stats, val_stats = instance_counts['train'], instance_counts['val']
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
            import ipdb;
            ipdb.set_trace()
            raise Exception

        instance_counts_per_class = instance_counts.sum(axis=0)
        image_counts_per_class = (instance_counts > 0).sum(axis=0)
        num_cars_per_image = instance_counts[:, semantic_class_names.index('car')]
        assert int(num_cars_per_image.max()) == num_cars_per_image.max()
        num_images_with_this_many_cars = [np.sum(num_cars_per_image == x) for x in range(int(num_cars_per_image.max()))]

        plt.figure(1);
        plt.clf()
        metric_name = split + '_' + 'instance_counts_per_class'
        metric_values = instance_counts_per_class
        labels = [s + '={}'.format(v) for s, v in zip(semantic_class_names, metric_values)]
        pie_chart(metric_values, labels)
        plt.title(metric_name)
        display_pyutils.save_fig_to_workspace(metric_name + '.png')

        plt.figure(2);
        plt.clf()
        metric_name = split + '_' + 'image_counts_per_class'
        metric_values = image_counts_per_class
        labels = [s + '={}'.format(v) for s, v in zip(semantic_class_names, metric_values)]
        pie_chart(metric_values, labels)
        plt.title(metric_name)
        display_pyutils.save_fig_to_workspace(metric_name + '.png')

        plt.figure(3)
        plt.clf()
        metric_name = split + '_' + 'num_images_with_this_many_cars'
        metric_values = num_images_with_this_many_cars
        labels = ['{} cars={}'.format(num, v) for num, v in enumerate(metric_values)]
        pie_chart(metric_values, labels)
        plt.title(metric_name)
        display_pyutils.save_fig_to_workspace(metric_name + '.png')


def pie_chart(values, labels, autopct='%1.1f%%'):
    explode = [0.1 * int(val != max(values)) for i, val in enumerate(values)]  # explode largest slice
    # Plot
    colors = [display_pyutils.GOOD_COLORS[np.mod(i, len(display_pyutils.GOOD_COLORS))] for i in range(len(values))]
    patches, text, _  = plt.pie(values, explode=explode, labels=labels, colors=colors,
                                autopct=autopct, shadow=True, startangle=140)
    plt.axis('equal')

    sort_legend = True
    if sort_legend:
        patches, labels, dummy = zip(*sorted(zip(patches, labels, values),
                                             key=lambda x: x[2],
                                             reverse=True))

    plt.legend(patches, labels, loc='center left', fontsize=8, bbox_to_anchor=(-0.02, 0.5))
    plt.tight_layout(pad=1.2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, required=True)
    parser.add_argument('--resume', help='Checkpoint path')
    args = parser.parse_args()

    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    train_stats, val_stats, default_train_dataset, default_val_dataset = \
        get_instance_counts('cityscapes')
    write_stats(train_stats, val_stats, default_train_dataset, default_val_dataset)


if __name__ == '__main__':
    main()
