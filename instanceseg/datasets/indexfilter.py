import numpy as np
import torch

from instanceseg.datasets import sampler
from instanceseg.utils.misc import pairwise_and, pairwise_or


class ValidIndexFilter(object):
    """
    Filters image list (implied by size of stats array) based on dataset statistics.
    Caches valid_indices.
    """

    def __init__(self, sampler_config: sampler.SamplerConfig, instance_counts: torch.Tensor = None,
                 semantic_class_pixel_counts: torch.Tensor = None):
        self.sampler_config = sampler_config
        self.instance_counts = instance_counts
        self.semantic_class_pixel_counts = semantic_class_pixel_counts

        if sampler_config.sem_cls_filter is not None:
            assert semantic_class_pixel_counts is not None, 'semantic_class_pixel_counts must be precomputed to ' \
                                                            'filter by semantic classes'
        if sampler_config.n_instances_ranges is not None:
            assert instance_counts is not None, 'instance_counts must be precomputed to filter by # instances'

        self._valid_indices = None

    def clear_cache(self):
        self._valid_indices = None

    @property
    def n_original_images(self):
        return self.instance_counts.shape[0] if self.instance_counts is not None else None

    @property
    def valid_indices(self):
        if self._valid_indices is None:
            n_instance_ranges = self.sampler_config.n_instances_ranges
            sem_cls_filter = self.sampler_config.sem_cls_filter
            n_images = self.sampler_config.n_images
            self._valid_indices = self.get_valid_indices(sem_cls_filter, n_instance_ranges, n_images)
        return self._valid_indices

    def get_valid_indices(self, sem_cls_filter, n_instance_ranges, n_images):
        if sem_cls_filter is None and n_instance_ranges is None and n_images is None:
            valid_indices = None  # 'all'
        else:
            valid_indices = self.valid_indices_from_instance_stats_config(sem_cls_filter, n_instance_ranges, union=False)
            valid_indices = self.subsample_n_valid_images(valid_indices, n_images)
        return valid_indices

    def get_valid_indices_single_sem_cls(self, single_sem_cls, n_instances_range):
        valid_indices = [True for _ in range(self.n_original_images)]
        if n_instances_range is not None:
            valid_indices = pairwise_and(valid_indices,
                                         self.filter_images_by_instance_range_from_counts(
                                             self.instance_counts, n_instances_range, [single_sem_cls]))
        elif single_sem_cls is not None:
            valid_indices = pairwise_and(valid_indices,
                                         self.filter_images_by_semantic_classes(self.semantic_class_pixel_counts,
                                                                                [single_sem_cls]))
        return valid_indices

    @staticmethod
    def filter_images_by_semantic_classes(semantic_class_pixel_counts, semantic_classes):
        valid_indices = []
        for image_idx in range(semantic_class_pixel_counts.shape[0]):
            is_valid = semantic_class_pixel_counts[:, semantic_classes].sum() > 0
            valid_indices.append(bool(is_valid))
        if sum(valid_indices) == 0:
            print(Warning('Found no valid images'))
        return valid_indices

    def valid_indices_from_instance_stats_config(self, semantic_class_vals, n_instance_ranges, union=False):
        """
        Example:
                semantic_classes = [15, 16]
                n_instance_ranges = [(2,4), (None,3)]
                union = False

                Result: boolean array where True represents an image that has 2 or 3 instances of semantic class 15
                and fewer than 3 instances of semantic class 16.

                If union = True, finds union of these images instead.
        """
        if n_instance_ranges is None and semantic_class_vals is None:
            valid_indices = [True for _ in range(self.n_original_images)]
        else:
            if len(n_instance_ranges) == 2:  # Check if just one range provided.
                assert isinstance(n_instance_ranges[0], (int, float)) and isinstance(n_instance_ranges[1], (int, float))
                n_instance_ranges = [n_instance_ranges]
            assert len(semantic_class_vals) == len(n_instance_ranges)
            try:
                assert not isinstance(n_instance_ranges[0], int)
                assert all([i is None or len(i) == 2 for i in n_instance_ranges])
            except AssertionError:
                raise Exception('There must be {} tuples and/or NoneTypes assigned to n_instances to match the number '
                                'of semantic classes.'.format(len(semantic_class_vals)))
            pairwise_combine = pairwise_and if not union else pairwise_or
            valid_indices = None
            for sem_cls, n_instances_range in zip(semantic_class_vals, n_instance_ranges):
                valid_indices_single_set = self.get_valid_indices_single_sem_cls(single_sem_cls=sem_cls,
                                                                                 n_instances_range=n_instances_range)
                if valid_indices is None:
                    valid_indices = valid_indices_single_set
                else:
                    valid_indices = pairwise_combine(valid_indices, valid_indices_single_set)
        return valid_indices

    @staticmethod
    def subsample_n_valid_images(valid_indices, n_images):
        if n_images is None:
            return valid_indices
        else:
            if sum(valid_indices) < n_images:
                raise Exception('Too few images to sample {}.  Choose a smaller value for n_images '
                                'in the sampler config, or change your filtering requirements for '
                                'the sampler.'.format(n_images))
            # Subsample n_images
            n_images_chosen = 0
            for idx in np.random.permutation(len(valid_indices)):
                if valid_indices[idx]:
                    if n_images_chosen == n_images:
                        valid_indices[idx] = False
                    else:
                        n_images_chosen += 1
            assert sum(valid_indices) == n_images
        return valid_indices

    @staticmethod
    def filter_images_by_non_bground(dataset, bground_val=0, void_val=-1):
        valid_indices = []
        for index, (img, (sem_lbl, _)) in enumerate(dataset):
            is_valid = (torch.sum(sem_lbl == bground_val) + torch.sum(sem_lbl == void_val)) != torch.numel(sem_lbl)
            if is_valid:
                valid_indices.append(True)
            else:
                valid_indices.append(False)
        if len(valid_indices) == 0:
            import ipdb;
            ipdb.set_trace()
            raise Exception('Found no valid images')
        return valid_indices

    @staticmethod
    def filter_images_by_instance_range_from_counts(instance_counts, n_instances_range=None, semantic_classes=None):
        """
        n_instances_range: (min, max+1), where value is None if you don't want to bound that direction
            -- default None is equivalent to (None, None) (All indices are valid.)
        python "range" rules -- [n_instances_min, n_instances_max)
        """
        if n_instances_range is None:
            return [True for _ in range(instance_counts.size(0))]

        assert len(n_instances_range) == 2, ValueError('range must be a tuple of (min, max).  You can set None for '
                                                       'either end of that range.')
        n_instances_min, n_instances_max = n_instances_range
        has_at_least_n_instances = instance_counts >= n_instances_min
        has_at_most_n_instances = instance_counts < n_instances_max \
            if n_instances_max is not None else torch.ByteTensor(instance_counts.size()).fill_(1)
        if semantic_classes is None:
            valid_indices_as_tensor = has_at_least_n_instances.sum(dim=1) * has_at_most_n_instances.sum(dim=1)
        else:
            valid_indices_as_tensor = sum([has_at_least_n_instances[:, sem_cls] for sem_cls in semantic_classes]) * \
                                      sum([has_at_most_n_instances[:, sem_cls] for sem_cls in semantic_classes])
        valid_indices = [x for x in valid_indices_as_tensor]
        if len(valid_indices) == 0:
            print(Warning('Found no valid images'))
        assert len(valid_indices) == instance_counts.size(0)
        return valid_indices