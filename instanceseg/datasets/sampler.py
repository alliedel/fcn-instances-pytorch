import torch
from torch.utils.data import sampler


def sampler_factory(sequential, index_weights=None, bool_index_subset=None):
    """
    sequential: False -- will shuffle the images.  True -- will return the same order of images for each call
    weights: weights to assign to each image
    bool_filter: True for each index you'd like to include in the sampler
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


class SamplerConfig(object):
    def __init__(self, n_images=None, sem_cls_filter=None, n_instances_range=None):
        self.n_images = n_images
        self.sem_cls_filter = sem_cls_filter
        self.n_instances_range = n_instances_range

    def __str__(self):
        return str(self.__dict__)
