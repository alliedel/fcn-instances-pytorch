import os

import numpy as np
import torch

from torchfcn.datasets import samplers, dataset_statistics


def get_sampler(dataset_instance_stats, sequential, sem_cls=None, n_instances_range=None, n_images=None):
    valid_indices = [True for _ in range(len(dataset_instance_stats.dataset))]
    if n_instances_range is not None:
        valid_indices = pairwise_and(valid_indices,
                                     dataset_instance_stats.filter_images_by_n_instances(n_instances_range, sem_cls))
    elif sem_cls is not None:
        valid_indices = pairwise_and(valid_indices, dataset_instance_stats.filter_images_by_semantic_classes(sem_cls))
    if n_images is not None:
        if sum(valid_indices) < n_images:
            raise Exception('Too few images to sample {}.  Choose a smaller value for n_images in the sampler '
                            'config, or change your filtering requirements for the sampler.'.format(n_images))

        # Subsample n_images
        n_images_chosen = 0
        for idx in np.random.permutation(len(valid_indices)):
            if valid_indices[idx]:
                if n_images_chosen == n_images:
                    valid_indices[idx] = False
                else:
                    n_images_chosen += 1
        try:
            assert sum(valid_indices) == n_images
        except AssertionError:
            import ipdb
            ipdb.set_trace()
            raise
    sampler = samplers.sampler_factory(sequential=sequential, bool_index_subset=valid_indices)(
        dataset_instance_stats.dataset)

    return sampler


def get_configured_sampler(dataset_type, dataset, sequential, n_instances_range, n_images, sem_cls_filter,
                           instance_count_file):
    if n_instances_range is not None:
        if dataset_type != 'voc':
            raise NotImplementedError('Need an established place to save instance counts')
        instance_counts = torch.from_numpy(np.load(instance_count_file)) \
            if os.path.isfile(instance_count_file) else None
        stats = dataset_statistics.InstanceDatasetStatistics(dataset, instance_counts)
        if instance_counts is None:
            stats.compute_statistics()
            instance_counts = stats.instance_counts
            np.save(instance_count_file, instance_counts.numpy())
    else:
        stats = dataset_statistics.InstanceDatasetStatistics(dataset)

    my_sampler = get_sampler(stats, sequential=sequential, n_instances_range=n_instances_range, sem_cls=sem_cls_filter,
                             n_images=n_images)
    if n_images:
        assert len(my_sampler.indices) == n_images
    return my_sampler


def pairwise_and(list1, list2):
    return [a and b for a, b in zip(list1, list2)]