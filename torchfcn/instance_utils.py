import torch
import numpy as np


class InstanceProblemConfig(object):
    """
    Used for both models and datasets to lay out the assumptions of each during instance
    segmentation.

    For models: Specifies the max # of instances of each semantic class it will attempt to produce.

    For datasets: Specifies the max # of instances of each class that appear in the
    (training) dataset.
    """

    def __init__(self, n_instances_by_semantic_id, void_value=-1, semantic_vals=None):
        assert len(semantic_vals) == len(n_instances_by_semantic_id)
        self.semantic_vals = semantic_vals or range(len(n_instances_by_semantic_id))
        self.void_value = void_value
        self.n_instances_by_semantic_id = n_instances_by_semantic_id

        # Compute derivative stuff

        self.sem_ids_by_instance_id = [id_into_sem_vals for
                                       id_into_sem_vals, n_inst in
                                       enumerate(self.n_instances_by_semantic_id)
                                       for _ in range(n_inst)]
        self.n_semantic_classes = len(self.semantic_vals)
        self.n_classes = sum(self.n_instances_by_semantic_id)
        self.semantic_instance_class_list = get_semantic_instance_class_list(
            n_instances_by_semantic_id)
        self.instance_to_semantic_mapping_matrix = get_instance_to_semantic_mapping(
            n_instances_by_semantic_id)


def get_semantic_instance_class_list(n_instances_by_semantic_id):
    return [sem_cls for sem_cls, n_instances in enumerate(n_instances_by_semantic_id)
            for _ in range(n_instances)]


def get_instance_to_semantic_mapping(n_instances_by_semantic_id,
                                     as_numpy=False):
    """
    returns a binary matrix, where semantic_instance_mapping is N x S
    (N = # instances, S = # semantic classes)
    semantic_instance_mapping[inst_idx, :] is a one-hot vector,
    and semantic_instance_mapping[inst_idx, sem_idx] = 1 iff that instance idx is an instance
    of that semantic class.
    """
    n_instance_classes = sum(n_instances_by_semantic_id)
    n_semantic_classes = len(n_instances_by_semantic_id)
    semantic_instance_class_list = get_semantic_instance_class_list(
        n_instances_by_semantic_id)
    instance_to_semantic_mapping_matrix = torch.zeros(
        (n_instance_classes, n_semantic_classes)).float()

    for instance_idx, semantic_idx in enumerate(semantic_instance_class_list):
        instance_to_semantic_mapping_matrix[instance_idx,
                                            semantic_idx] = 1
    return instance_to_semantic_mapping_matrix if not as_numpy else \
        instance_to_semantic_mapping_matrix.numpy()


def get_n_instances_by_semantic_id(semantic_ids, instance_ids, background_ids):
    all_train_ids = semantic_ids + instance_ids + background_ids
    max_train_id = max(all_train_ids)

