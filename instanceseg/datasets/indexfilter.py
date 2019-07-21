import numpy as np
import torch
import logging

from instanceseg.datasets import sampler
from instanceseg.utils.misc import pairwise_and, pairwise_or

logger = logging.getLogger(__name__)


class ValidIndexFilter(object):
    """
    Filters image list (implied by size of stats array) based on dataset statistics.
    Caches valid_indices.

    Note n_original_images is required, even though that information might be contained in
    instance_counts / semantic_class_pixel_counts.
    """

    def __init__(self, sampler_config: sampler.SamplerConfig, n_original_images: int,
                 instance_counts: torch.Tensor = None,
                 semantic_class_pixel_counts: torch.Tensor = None,
                 occlusion_counts: torch.Tensor = None):
        self.sampler_config = sampler_config
        self.instance_counts = instance_counts
        self.semantic_class_pixel_counts = semantic_class_pixel_counts
        self.n_original_images = n_original_images
        self.occlusion_counts = occlusion_counts

        if sampler_config.requires_semantic_pixel_counts:
            assert semantic_class_pixel_counts is not None, 'semantic_class_pixel_counts must be ' \
                                                            'precomputed to filter by semantic ' \
                                                            'classes'
        if sampler_config.requires_instance_counts:
            assert instance_counts is not None, \
                'instance_counts must be precomputed to filter by # instances'

        if sampler_config.requires_occlusion_counts:
            assert occlusion_counts is not None, 'occlusion_counts must be precomputed to filter ' \
                                                 'by # instances'

        self._valid_indices = None

    def clear_cache(self):
        self._valid_indices = None

    @property
    def valid_indices(self):
        if self._valid_indices is None:
            n_instance_ranges = self.sampler_config.n_instances_ranges
            sem_cls_filter = self.sampler_config.sem_cls_filter_values
            n_images = self.sampler_config.n_images
            n_occlusions_range = self.sampler_config.n_occlusions_range
            self._valid_indices = self.valid_indices_from_stats_config(
                sem_cls_filter, n_instance_ranges, n_occlusions_range, n_images)
        return self._valid_indices

    def valid_indices_from_stats_config(self, semantic_class_vals, n_instance_ranges,
                                        n_occlusions_range, n_images, union=False):
        """
        Example:
                semantic_classes = [15, 16]
                n_instance_ranges = [(2,4), (None,3)]
                union = False

                Result: boolean array where True represents an image that has 2 or 3 instances of
                semantic class 15 and fewer than 3 instances of semantic class 16.

                If union = True, finds union of these images instead.
        """
        pairwise_combine = pairwise_and if not union else pairwise_or

        valid_indices = [True for _ in range(self.n_original_images)]

        if n_instance_ranges is not None:
            for sem_cls, n_instances_range in zip(semantic_class_vals, n_instance_ranges):
                valid_indices_single_set = self.get_valid_indices_single_sem_cls_instances(
                    single_sem_cls=sem_cls,
                    n_instances_range=n_instances_range)
                valid_indices = pairwise_combine(valid_indices, valid_indices_single_set)
        if n_occlusions_range is not None:
            valid_indices = pairwise_combine(
                valid_indices, self.get_valid_indices_total_occlusions(
                    sem_cls_vals=semantic_class_vals, valid_occlusion_range=n_occlusions_range))
        if n_images is not None:
            valid_indices = self.subsample_n_valid_images(valid_indices, n_images)
        return valid_indices

    def get_valid_indices_total_occlusions(self, sem_cls_vals, valid_occlusion_range):
        counts = self.occlusion_counts
        totals_per_img_for_select_classes = counts[:, sem_cls_vals].sum(dim=1)
        valid_indices = self.get_valid_indices_from_scalar_stat(
            totals_per_img_for_select_classes, valid_occlusion_range)
        return valid_indices

    def get_valid_indices_single_sem_cls_instances(self, single_sem_cls, n_instances_range):
        valid_indices = [True for _ in range(self.n_original_images)]
        counts = self.instance_counts
        valid_range = n_instances_range
        if valid_range is not None:
            assert counts is not None
            valid_indices = pairwise_and(valid_indices,
                                         self.filter_images_by_range_from_counts2d(
                                             counts, valid_range,
                                             [single_sem_cls]))
        elif single_sem_cls is not None:
            valid_indices = pairwise_and(valid_indices,
                                         self.filter_images_by_semantic_classes(
                                             self.semantic_class_pixel_counts,
                                             [single_sem_cls]))
        return valid_indices

    @staticmethod
    def filter_images_by_semantic_classes(semantic_class_pixel_counts, semantic_classes):
        valid_indices = []
        for image_idx in range(semantic_class_pixel_counts.shape[0]):
            is_valid = semantic_class_pixel_counts[:, semantic_classes].sum() > 0
            valid_indices.append(bool(is_valid))
        if sum(valid_indices) == 0:
            logger.warning('Found no valid images')
        return valid_indices

    @staticmethod
    def subsample_n_valid_images(valid_indices, n_images):
        if n_images is None:
            return valid_indices
        else:
            if sum(valid_indices) < n_images:
                raise Exception('Too few images ({}) to sample {}.  Choose a smaller value for n_images '
                                'in the sampler config, or change your filtering requirements for '
                                'the sampler.'.format(sum(valid_indices), n_images))
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
            is_valid = (torch.sum(sem_lbl == bground_val) + torch.sum(
                sem_lbl == void_val)) != torch.numel(sem_lbl)
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
    def get_valid_indices_from_scalar_stat(vec_stat_per_img, valid_range):
        n_valid_min, n_valid_max = valid_range
        has_at_least_n_min = (vec_stat_per_img >= n_valid_min) \
            if n_valid_min is not None else torch.ByteTensor(vec_stat_per_img.size()).fill_(1)
        has_at_most_n_max = (vec_stat_per_img < n_valid_max) \
            if n_valid_max is not None else torch.ByteTensor(vec_stat_per_img.size()).fill_(1)
        valid_indices_as_tensor = has_at_least_n_min * has_at_most_n_max
        valid_indices = [x > 0 for x in valid_indices_as_tensor.tolist()]  # list of bool
        return valid_indices

    @staticmethod
    def filter_images_by_range_from_counts2d(counts, valid_range=None, semantic_classes=None):
        """
        valid_range: (min, max+1), where value is None if you don't want to bound that direction
            -- default None is equivalent to (None, None) (All indices are valid.)
        python "range" rules -- [n_min, n_max)
        """
        if valid_range is None:
            return [True for _ in range(counts.size(0))]

        assert len(valid_range) == 2, ValueError(
            'range must be a tuple of (min, max).  You can set None for '
            'either end of that range.')
        n_valid_min, n_valid_max = valid_range
        has_at_least_n_min = (counts >= n_valid_min) \
            if n_valid_min is not None else torch.ByteTensor(counts.size()).fill_(1)
        has_at_most_n_max = (counts < n_valid_max) \
            if n_valid_max is not None else torch.ByteTensor(counts.size()).fill_(1)
        if semantic_classes is None:
            valid_indices_as_tensor = has_at_least_n_min.sum(
                dim=1) * has_at_most_n_max.sum(dim=1)
        else:
            valid_indices_as_tensor = sum(
                [has_at_least_n_min[:, sem_cls] for sem_cls in semantic_classes]) * \
                                      sum([has_at_most_n_max[:, sem_cls] for sem_cls in
                                           semantic_classes])
        valid_indices = [x for x in valid_indices_as_tensor]
        if len(valid_indices) == 0:
            logger.warning('Found no valid images')
        assert len(valid_indices) == counts.size(0)
        return valid_indices
