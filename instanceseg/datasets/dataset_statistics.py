import numpy as np
import torch
import tqdm
import cv2
import abc
import os
import logging

logger = logging.getLogger(__name__)


class DatasetStatisticCacheInterface(object):
    """
    Stores some statistic about a set of images contained in a Dataset class (e.g. - number of instances in the image).
    Inheriting from this handles some of the save/load/caching.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, cache_file=None, override=False):
        self._stat_tensor = None
        self.cache_file = cache_file
        self.override = override

    @abc.abstractmethod
    def labels(self):
        raise NotImplementedError

    @property
    def shape(self):
        return self.stat_tensor.shape

    @property
    def n_images(self):
        return self.shape[0]

    @staticmethod
    def load(stats_filename):
        return torch.from_numpy(np.load(stats_filename))

    @staticmethod
    def save(statistics, stats_filename):
        np.save(stats_filename, statistics)

    @property
    def stat_tensor(self):
        if self._stat_tensor is None:
            raise Exception('Statistic has not yet been computed.  Run {}.compute(<dataset>)'.format(
                self.__class__.__name__))
        return self._stat_tensor

    def compute_or_retrieve(self, dataset):
        if self.cache_file is None:
            logger.info('Computing statistics without cache')
            self._stat_tensor = self._compute(dataset)
        elif self.override or not os.path.exists(self.cache_file):
            logger.info('Computing statistics for file {}'.format(self.cache_file))
            self._stat_tensor = self._compute(dataset)
            self.save(self._stat_tensor, self.cache_file)
        else:
            logger.info('Loading statistics from file {}'.format(self.cache_file))
            self._stat_tensor = self.load(self.cache_file)

    @abc.abstractmethod
    def _compute(self, dataset):
        raise NotImplementedError


class PixelsPerSemanticClass(DatasetStatisticCacheInterface):

    def __init__(self, semantic_class_vals, semantic_class_names=None, cache_file=None,
                 override=False):
        super(PixelsPerSemanticClass, self).__init__(cache_file, override)
        self.semantic_class_names = semantic_class_names or ['{}'.format(v) for v in
                                                             semantic_class_vals]
        self.semantic_class_vals = semantic_class_vals

    @property
    def labels(self):
        return self.semantic_class_names

    def _compute(self, dataset):
        # semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
        semantic_pixel_counts = self.compute_semantic_pixel_counts(dataset, self.semantic_class_vals)
        self._stat_tensor = semantic_pixel_counts
        return semantic_pixel_counts

    @staticmethod
    def compute_semantic_pixel_counts(dataset, semantic_class_vals):
        semantic_pixel_counts_nested_list = []
        for idx, (img, (sem_lbl, inst_lbl)) in tqdm.tqdm(
                enumerate(dataset), total=len(dataset),
                desc='Running semantic pixel statistics on dataset'.format(dataset), leave=False):
            semantic_pixel_counts_nested_list.append([(sem_lbl == sem_val).sum() for sem_val in \
                                                      semantic_class_vals])
        semantic_pixel_counts = torch.IntTensor(semantic_pixel_counts_nested_list)
        return semantic_pixel_counts


class NumberofInstancesPerSemanticClass(DatasetStatisticCacheInterface):
    """
    Computes NxS nparray: For each of N images, contains the number of instances of each of S semantic classes
    """

    def __init__(self, semantic_classes, cache_file=None, override=False):
        super(NumberofInstancesPerSemanticClass, self).__init__(cache_file, override)
        self.semantic_classes = semantic_classes

    @property
    def labels(self):
        return self.semantic_classes

    def _compute(self, dataset):
        # semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
        instance_counts = self.compute_instance_counts(dataset, self.semantic_classes)
        self._stat_tensor = instance_counts
        return instance_counts

    @staticmethod
    def compute_instance_counts(dataset, semantic_classes):
        instance_counts = torch.ones(len(dataset), len(semantic_classes)) * -1
        for idx, (img, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(dataset), total=len(dataset),
                                                         desc='Running instance statistics on dataset'.format(dataset),
                                                         leave=False):
            for sem_idx, sem_val in enumerate(semantic_classes):
                sem_locations_bool = sem_lbl == sem_val
                if torch.sum(sem_locations_bool) > 0:
                    my_max = inst_lbl[sem_locations_bool].max()
                    instance_counts[idx, sem_idx] = my_max
                else:
                    instance_counts[idx, sem_idx] = 0
                if sem_idx == 0 and instance_counts[idx, sem_idx] > 0:
                    import ipdb
                    ipdb.set_trace()
                    raise Exception('inst_lbl should be 0 wherever sem_lbl is 0')
        return instance_counts


def compute_occlusion_counts(dataset, semantic_classes=None):
    semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
    occlusion_counts = torch.ones(len(dataset), len(semantic_classes), len(semantic_classes)) * 0
    for idx, (img, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(dataset), total=len(dataset),
                                                     desc='Running occlusion statistics on dataset'.format(dataset),
                                                     leave=False):

        # Get overlapping instances of same semantic class
        inst_lbl_dilated = inst_lbl.copy()
        for sem_idx, sem_val in enumerate(semantic_classes):
            n_occlusions_this_sem_cls = 0
            sem_locations_bool = sem_lbl == sem_val
            if torch.sum(sem_locations_bool) > 1:  # needs to be more than one instance to have occlusion with same
                # class.
                num_instances = inst_lbl[sem_locations_bool].max()
                if num_instances > 0:
                    intermediate_union = sem_locations_bool
                    for inst_val in range(num_instances):
                        dilated_instance_mask = cv2.dilate(inst_lbl == inst_val, kernel=np.ones((3, 3)), iterations=2)
                        intersections_with_previous_instances = (sem_locations_bool * dilated_instance_mask *
                                                                 intermediate_union).int()
                        intermediate_union += (sem_locations_bool * (inst_lbl == inst_val)).int()
                        amt_of_overlap = intersections_with_previous_instances.sum()
                        n_occlusions_this_sem_cls += int(amt_of_overlap > 0)
                    cv2.dilate(inst_lbl, kernel=np.ones((3, 3)), dst=inst_lbl_dilated, iterations=1)
            else:
                occlusion_counts[idx, sem_idx] = 0

        # Get overlapping instances of any semantic class

    return occlusion_counts
