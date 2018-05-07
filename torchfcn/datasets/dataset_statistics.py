import numpy as np
import torch
import tqdm


class InstanceDatasetStatistics(object):
    def __init__(self, dataset):
        self.dataset = dataset
        self.instance_counts = None
        self.non_bground_images = None

    def compute_statistics(self):
        instance_counts = self.compute_instance_counts()
        self.instance_counts = instance_counts
        self.non_bground_img_idxs = self.filter_images_by_non_bground(self.dataset)

    def compute_instance_counts(self, semantic_classes=None):
        dataset = self.dataset
        instance_counts = compute_instance_counts(dataset, semantic_classes)
        return instance_counts

    def filter_images_by_semantic_classes(self, semantic_classes):
        valid_indices = filter_images_by_semantic_classes(self.dataset, semantic_classes)
        return valid_indices

    def filter_images_by_n_instances(self, n_instances, semantic_classes=None):
        valid_indices = filter_images_by_n_instances(self.dataset, n_instances, semantic_classes)
        return valid_indices

    def filter_images_by_non_bground(self, bground_val=0, void_val=-1):
        valid_indices = filter_images_by_non_bground(self.dataset, bground_val, void_val)
        return valid_indices


def filter_images_by_semantic_classes(dataset, semantic_classes):
    valid_indices = []
    for index, (img, (sem_lbl, _)) in enumerate(dataset):
        is_valid = torch.sum([(sem_lbl == sem_val) for sem_val in semantic_classes])
        if is_valid:
            valid_indices.append(index)
    if len(valid_indices) == 0:
        print(Warning('Found no valid images'))


def filter_images_by_n_instances(dataset, n_instances, semantic_classes=None):
    valid_indices = []
    for index, (img, (sem_lbl, inst_lbl)) in enumerate(dataset):
        if semantic_classes is None:
            is_valid = inst_lbl.max()[0] >= n_instances
        else:
            max_per_class = [inst_lbl[sem_lbl == sem_cls].max() for sem_cls in semantic_classes]
            is_valid = any([m > n_instances for m in max_per_class])
        if is_valid:
            valid_indices.append(index)
    if len(valid_indices) == 0:
        print(Warning('Found no valid images'))


def compute_instance_counts(dataset, semantic_classes):
    semantic_classes = semantic_classes or range(dataset.n_semantic_classes)
    instance_counts = np.ones((len(dataset), len(semantic_classes))) * np.nan
    for idx, (img, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(dataset), total=len(dataset),
                                                     desc='Analyzing VOC files', ncols=80, leave=False):
        for sem_idx, sem_val in enumerate(semantic_classes):
            sem_locations_bool = sem_lbl == sem_val
            if torch.np.any(sem_locations_bool):
                instance_counts[idx, sem_idx] = inst_lbl[sem_locations_bool].max()
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
            valid_indices.append(index)
    if len(valid_indices) == 0:
        import ipdb; ipdb.set_trace()
        raise Exception('Found no valid images')
    return valid_indices
