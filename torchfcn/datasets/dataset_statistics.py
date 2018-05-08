import torch
import tqdm


class InstanceDatasetStatistics(object):
    def __init__(self, dataset, instance_counts=None):
        self.dataset = dataset
        self.instance_counts = instance_counts
        self.non_bground_images = None

    def compute_statistics(self):
        instance_counts = self.compute_instance_counts()
        self.instance_counts = instance_counts

    def compute_instance_counts(self, semantic_classes=None):
        dataset = self.dataset
        instance_counts = compute_instance_counts(dataset, semantic_classes)
        return instance_counts

    def filter_images_by_semantic_classes(self, semantic_classes):
        valid_indices = filter_images_by_semantic_classes(self.dataset, semantic_classes)
        return valid_indices

    def filter_images_by_n_instances(self, n_instances_range=None, semantic_classes=None):
        if self.instance_counts is None:
            self.compute_statistics()
        valid_indices = filter_images_by_n_instances_from_counts(self.instance_counts, n_instances_range,
                                                                 semantic_classes)
        return valid_indices

    def filter_images_by_non_bground(self, bground_val=0, void_val=-1):
        valid_indices = filter_images_by_non_bground(self.dataset, bground_val, void_val)
        return valid_indices


def filter_images_by_semantic_classes(dataset, semantic_classes):
    valid_indices = []
    for index, (img, (sem_lbl, _)) in enumerate(dataset):
        is_valid = sum([(sem_lbl == sem_val).sum() for sem_val in semantic_classes])
        if is_valid:
            valid_indices.append(True)
        else:
            valid_indices.append(False)
    if sum(valid_indices) == 0:
        print(Warning('Found no valid images'))
    return valid_indices


def max_or_default(tensor, default_val=0):
    return default_val if torch.numel(tensor) == 0 else torch.max(tensor)


def filter_images_by_n_instances_from_counts(instance_counts, n_instances_range=None,
                                             semantic_classes=None):
    """
    n_instances_range: (min, max+1), where value is None if you don't want to bound that direction
        -- default None is equivalent to (None, None) (All indices are valid.)
    python "range" rules -- [n_instances_min, n_instances_max)
    """
    if n_instances_range is None:
        return [True for _ in range(instance_counts.size(0))]

    assert len(n_instances_range) == 2, ValueError('range must be a tuple of (min, max).  You can set None for either '
                                                   'end of that range.')
    n_instances_min, n_instances_max = n_instances_range
    has_at_least_n_instances = instance_counts >= n_instances_min
    if n_instances_max is not None:
        has_at_most_n_instances = instance_counts < n_instances_max
    else:
        has_at_most_n_instances = torch.ones_like(instance_counts)
    if semantic_classes is None:
        valid_indices_as_tensor = has_at_least_n_instances.sum(dim=1) * has_at_most_n_instances.sum(dim=1)
    else:
        valid_indices_as_tensor = sum([has_at_least_n_instances[:, sem_cls] for sem_cls in semantic_classes]) *\
                                  sum([has_at_most_n_instances[:, sem_cls] for sem_cls in semantic_classes])
    valid_indices = [x for x in valid_indices_as_tensor]
    if len(valid_indices) == 0:
        print(Warning('Found no valid images'))
    assert len(valid_indices) == instance_counts.size(0)
    return valid_indices


def compute_instance_counts(dataset, semantic_classes=None):
    semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
    instance_counts = torch.ones(len(dataset), len(semantic_classes)) * torch.np.nan
    for idx, (img, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(dataset), total=len(dataset),
                                                     desc='Analyzing VOC files', ncols=80, leave=False):
        for sem_idx, sem_val in enumerate(semantic_classes):
            sem_locations_bool = sem_lbl == sem_val
            if torch.sum(sem_locations_bool) > 0:
                my_max = inst_lbl[sem_locations_bool].max()
                instance_counts[idx, sem_idx] = my_max
            else:
                instance_counts[idx, sem_idx] = 0
            if sem_idx == 0 and instance_counts[idx, sem_idx] > 0:
                import ipdb; ipdb.set_trace()
                raise Exception('inst_lbl should be 0 wherever sem_lbl is 0')
    return instance_counts


def filter_images_by_non_bground(dataset, bground_val=0, void_val=-1):
    valid_indices = []
    for index, (img, (sem_lbl, _)) in enumerate(dataset):
        is_valid = (torch.sum(sem_lbl == bground_val) + torch.sum(sem_lbl == void_val)) != torch.numel(sem_lbl)
        if is_valid:
            valid_indices.append(True)
        else:
            valid_indices.append(False)
    if len(valid_indices) == 0:
        import ipdb; ipdb.set_trace()
        raise Exception('Found no valid images')
    return valid_indices
