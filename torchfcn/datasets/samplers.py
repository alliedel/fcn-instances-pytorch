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


# class SubsetRandomSampler(sampler.Sampler):
#     r"""Samples elements randomly from a given list of indices, without replacement.
#
#     Arguments:
#         indices (list): a list of indices
#     """
#
#     def __init__(self, indices):
#         self.indices = indices
#
#     def __iter__(self):
#         return (self.indices[i] for i in torch.randperm(len(self.indices)))
#
#     def __len__(self):
#         return len(self.indices)
#
#
#
# class WeightedRandomSampler(sampler.Sampler):
#     r"""Samples elements from [0,..,len(weights)-1] with given probabilities (weights).
#
#     Arguments:
#         weights (list)   : a list of weights, not necessary summing up to one
#         num_samples (int): number of samples to draw
#         replacement (bool): if ``True``, samples are drawn with replacement.
#             If not, they are drawn without replacement, which means that when a
#             sample index is drawn for a row, it cannot be drawn again for that row.
#     """
#
#     def __init__(self, weights, num_samples, replacement=True):
#         if not isinstance(num_samples, _int_classes) or isinstance(num_samples, bool) or \
#                 num_samples <= 0:
#             raise ValueError("num_samples should be a positive integeral "
#                              "value, but got num_samples={}".format(num_samples))
#         if not isinstance(replacement, bool):
#             raise ValueError("replacement should be a boolean value, but got "
#                              "replacement={}".format(replacement))
#         self.weights = torch.tensor(weights, dtype=torch.double)
#         self.num_samples = num_samples
#         self.replacement = replacement
#
#     def __iter__(self):
#         return iter(torch.multinomial(self.weights, self.num_samples, self.replacement))
#
#     def __len__(self):
#         return self.num_samples


