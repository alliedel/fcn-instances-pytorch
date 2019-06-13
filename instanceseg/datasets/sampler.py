import numpy as np
import torch
from torch.utils.data import sampler


class SamplerConfigWithoutValues(object):

    def __init__(self, n_images=None, sem_cls_filter_names=None, n_instances_ranges=None,
                 n_occlusions_range=None):
        self.n_images = n_images
        self.sem_cls_filter_names = sem_cls_filter_names
        self.n_instances_ranges = n_instances_ranges
        self.n_occlusions_range = n_occlusions_range

        if self.n_instances_ranges is not None:  # Check if just one range
            # provided (should be list of tuples, one per class)
            if self.is_valid_tuple(self.n_instances_ranges):
                self.n_instances_ranges = [self.n_instances_ranges for _ in
                                           range(len(self.sem_cls_filter_names))]

        self.assert_valid()

    @staticmethod
    def is_valid_tuple(instance_range):
        return len(instance_range) == 2 and all(
            SamplerConfig.is_valid_instance_limit_val(val) for val in
            instance_range)

    @staticmethod
    def is_valid_instance_limit_val(inst_lim_val):
        return (inst_lim_val is None) or isinstance(inst_lim_val, (int, float))

    def assert_valid(self):
        if self.sem_cls_filter_names is None:
            assert self.n_instances_ranges is None
        else:
            if self.n_instances_ranges is not None:
                assert len(self.sem_cls_filter_names) == len(self.n_instances_ranges)
                assert all([i is None or self.is_valid_tuple(i) for i in self.n_instances_ranges]), \
                    'Instance limits {} not valid'.format(self.n_instances_ranges)
            elif self.n_occlusions_range is None:
                print(self)
                raise Exception('Either n_occlusions_range or n_instances_ranges must be '
                                'specified when selecting semantic classes, even if they just '
                                'include all None.  Change this if it becomes annoying to be '
                                'explicit about it, but it will hopefully prevent unintentional '
                                'configurations.')

    def __str__(self):
        return str(self.__dict__)

    @property
    def requires_semantic_pixel_counts(self):
        if self.sem_cls_filter_names is not None:
            return True

    @property
    def requires_instance_counts(self):
        if self.n_instances_ranges is None:
            return False
        elif type(self.n_instances_ranges) is list:
            return not all(i is None for i in self.n_instances_ranges)
        else:
            return True

    @property
    def requires_occlusion_counts(self):
        if self.n_occlusions_range is None:
            return False
        elif type(self.n_occlusions_range) is list or tuple:
            assert len(self.n_occlusions_range) == 2
            return not all(o is None for o in self.n_occlusions_range)
        else:
            raise NotImplementedError


class SamplerConfig(SamplerConfigWithoutValues):
    def __init__(self, n_images=None, sem_cls_filter_names=None, n_instances_ranges=None,
                 semantic_class_names=None, n_occlusions_range=None):
        super(SamplerConfig, self).__init__(n_images=n_images,
                                            sem_cls_filter_names=sem_cls_filter_names,
                                            n_instances_ranges=n_instances_ranges,
                                            n_occlusions_range=n_occlusions_range)
        self.sem_cls_filter_values = convert_sem_cls_filter_from_names_to_values(
            self.sem_cls_filter_names, semantic_class_names)

    @classmethod
    def create_from_cfg_without_vals(cls, old_inst: SamplerConfigWithoutValues,
                                     semantic_class_names):
        return cls(n_images=old_inst.n_images, sem_cls_filter_names=old_inst.sem_cls_filter_names,
                   n_instances_ranges=old_inst.n_instances_ranges,
                   semantic_class_names=semantic_class_names,
                   n_occlusions_range=old_inst.n_occlusions_range)


def get_pytorch_sampler(sequential, index_weights=None, bool_index_subset=None):
    """
    sequential: False -- will shuffle the images.  True -- will return the same order of images for each call
    index_weights: weights to assign to each image
    bool_index_subset: True for each index you'd like to include in the sampler.  None if select all.
    """

    class SubsetWeightedSampler(sampler.Sampler):
        def __init__(self, datasource):
            if len(datasource) == 0:
                raise Exception('datasource must be nonempty')
            self.initial_indices = range(len(datasource))
            self.sequential = sequential
            self.indices = self.get_sample_indices_from_initial(self.initial_indices)

        def __iter__(self):
            if sequential:
                return iter(self.indices)
            else:
                return iter(self.indices[x] for x in torch.randperm(len(self.indices)).long())

        def __len__(self):
            return len(self.indices)

        def copy(self, sequential_override=None, cut_n_images=None):
            copy_of_self = SubsetWeightedSampler(self.initial_indices)
            copy_of_self.initial_indices = self.initial_indices
            copy_of_self.sequential = sequential_override or self.sequential
            copy_of_self.indices = self.indices
            if cut_n_images:
                copy_of_self.cut_sampler(cut_n_images)
            return copy_of_self

        def cut_sampler(self, n_images):
            self.indices = [self.indices[i] for i in range(n_images)]

        @classmethod
        def get_sample_indices_from_initial(cls, initial_indices):
            if index_weights is not None:
                raise NotImplementedError
            if bool_index_subset is not None:
                new_indices = [index for index in initial_indices if bool_index_subset[index]]
            else:
                new_indices = initial_indices
            return new_indices

    return SubsetWeightedSampler


def convert_sem_cls_filter_from_names_to_values(sem_cls_filter, semantic_class_names):
    if sem_cls_filter is None:
        return None
    try:
        sem_cls_filter_values = [semantic_class_names.index(class_name)
                                 for class_name in sem_cls_filter]
    except:
        sem_cls_filter_values = [int(np.where(semantic_class_names == class_name)[0][0])
                                 for class_name in sem_cls_filter]
    return sem_cls_filter_values
