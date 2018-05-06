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
