from typing import List

import numpy as np
import torch
import yaml
from torch import nn

from instanceseg.datasets.coco_format import CategoryCOCOFormat
from instanceseg.utils.misc import warn


class InstanceProblemConfig(object):
    """
    Used for both models and datasets to lay out the assumptions of each during instance
    segmentation.

    For models: Specifies the max # of instances of each semantic class it will attempt to produce.

    For datasets: Specifies the max # of instances of each class that appear in the
    (training) dataset.
    """

    def __init__(self, n_instances_by_semantic_id, labels_table: List[CategoryCOCOFormat] = None,
                 void_value=255, include_instance_channel0=False, map_to_semantic=False):
        """
        For semantic, include_instance_channel0=True
        n_instances_by_semantic_id = [0, 0, ..]
        """

        if labels_table is not None:
            assert len(labels_table) == len(n_instances_by_semantic_id)
            if any([l['id'] == void_value for l in labels_table]):
                void_ls = [l for l in labels_table if l['id'] == void_value]
                warn('We don\'t allow void value in labels_table.  Not considered a semantic class.  '
                     'Removing {} from labels table.'.format(['{} ({})'.format(l['name'], l['id']) for l in
                                                              void_ls]))
                labels_table = [l for l in labels_table if l['id'] != void_value]
        self.labels_table = labels_table
        assert n_instances_by_semantic_id is not None, ValueError
        self.map_to_semantic = map_to_semantic
        self.void_value = void_value
        self.include_instance_channel0 = include_instance_channel0
        self.n_instances_by_semantic_id = n_instances_by_semantic_id \
            if not map_to_semantic else [1 for _ in n_instances_by_semantic_id]
        self.model_n_instances_by_semantic_id = n_instances_by_semantic_id \
            if not map_to_semantic else [1 for _ in n_instances_by_semantic_id]

    @property
    def thing_class_names(self):
        return [l.name for l in self.labels_table if l.isthing]

    @property
    def stuff_class_names(self):
        return [l.name for l in self.labels_table if not l.isthing]

    @property
    def thing_class_vals(self):
        return [l.id for l in self.labels_table if l.isthing]

    @property
    def thing_class_ids(self):
        return [i for i, l in enumerate(self.labels_table) if l.isthing]

    @property
    def stuff_class_ids(self):
        return [i for i, l in enumerate(self.labels_table) if not l.isthing]

    @property
    def stuff_class_vals(self):
        return [l.id for l in self.labels_table if not l.isthing]

    @property
    def semantic_class_names(self):
        return [l.name for l in self.labels_table]

    @property
    def semantic_class_names_by_model_id(self):
        return {i: l.name for i, l in enumerate(self.labels_table)}

    @property
    def semantic_class_names_by_val(self):
        return {l.val: l.name for i, l in enumerate(self.labels_table)}

    @property
    def n_semantic_classes(self):
        return len(self.semantic_class_names)

    @property
    def semantic_colors(self):
        return [l.color for l in self.labels_table]

    @property
    def semantic_ids(self):
        return list(range(len(self.semantic_vals)))

    @property
    def semantic_vals(self):
        return [l.id for l in self.labels_table]

    @property
    def semantic_transformed_label_ids(self):
        return list(range(len(self.labels_table)))

    @property
    def has_instances(self):
        return [l.isthing for l in self.labels_table]

    @property
    def supercategories(self):
        return [l.supercategory for l in self.labels_table]

    @property
    def n_classes(self):
        return len(self.model_channel_semantic_ids) if not self.map_to_semantic else self.n_semantic_classes

    @property
    def model_channel_semantic_ids(self):
        """
        n_instances_by_semantic_id: [1, 3, 3, 3] => Returns [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
        """
        return np.array([sem_cls for sem_cls, n_channels in enumerate(self.n_instances_by_semantic_id)
                         for _ in range(n_channels)])

    @property
    def model_channel_semantic_vals(self):
        return [self.semantic_vals[i] for i in self.model_channel_semantic_ids]

    @property
    def state_dict(self):
        return dict(
            labels_table=self.labels_table,
            map_to_semantic=self.map_to_semantic,
            void_value=self.void_value,
            include_instance_channel0=self.include_instance_channel0,
            n_instances_by_semantic_id=self.n_instances_by_semantic_id
        )

    @property
    def instance_count_id_list(self):
        """
        Example:
            input: [0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
            non_instance_sem_classes=(0,)  # (background class gets inst channel label 0)
                Returns:
                    if include_channel0=False:
                        [0, 1, 2, 3, 1, 2, 3, 1, 2, 3]
                    if include_channel0=True:
                        [0, 0, 1, 2, 0, 1, 2, 0, 1, 2]

        """
        unique_semantic_classes = self.semantic_ids
        model_channel_semantic_ids = self.model_channel_semantic_ids
        inst_val_arr = np.empty((len(model_channel_semantic_ids),))
        for sem_cls in unique_semantic_classes:
            sem_cls_locs = model_channel_semantic_ids == sem_cls
            if sem_cls in list(self.stuff_class_ids):
                assert sum(sem_cls_locs) == 1
                inst_val_arr[sem_cls_locs] = 0
            elif sem_cls in list(self.thing_class_ids):
                inst_val_arr[sem_cls_locs] = np.arange(sem_cls_locs.sum()) + \
                                             (0 if self.include_instance_channel0 else 1)
            else:
                raise Exception('Bug: val should have been in things or stuff list')
        return inst_val_arr.astype(int).tolist()

    @property
    def instance_to_semantic_mapping_matrix(self, as_numpy=True):
        return get_instance_to_semantic_mapping_from_model_channel_semantic_ids(
            self.model_channel_semantic_ids, as_numpy)

    @property
    def instance_to_semantic_conv1x1(self):
        return nn.Conv2d(in_channels=len(self.model_channel_semantic_ids), out_channels=self.n_semantic_classes,
                         kernel_size=1, bias=False)

    @property
    def sem_ids_by_instance_id(self):
        return [id_into_sem_vals for id_into_sem_vals, n_inst in
                enumerate(self.n_instances_by_semantic_id) for _ in range(n_inst)]

    @classmethod
    def load(cls, yaml_path):
        args_state_dict = yaml.safe_load(open(yaml_path, 'r'))
        args_state_dict['labels_table'] = [CategoryCOCOFormat(**l) for l in args_state_dict['labels_table']]
        return cls(**args_state_dict)

    def save(self, yaml_path):
        state_dict = self.state_dict.copy()
        state_dict['labels_table'] = [l.__dict__ for l in state_dict['labels_table']]
        yaml.safe_dump(state_dict, open(yaml_path, 'w'))

    @staticmethod
    def _get_channel_labels(model_channel_semantic_ids, model_channel_instance_ids, class_names, map_to_semantic,
                            sem_inst_format):
        if class_names is None:
            model_channel_semantic_ids = model_channel_semantic_ids
        else:
            model_channel_semantic_ids = [class_names[c] for c in model_channel_semantic_ids]
        if map_to_semantic:
            channel_labels = [sem_inst_format.format(sem_cls, '') for sem_cls, inst_id in zip(
                model_channel_semantic_ids, model_channel_instance_ids)]
        else:
            channel_labels = [sem_inst_format.format(sem_cls, int(inst_id)) for sem_cls, inst_id in zip(
                model_channel_semantic_ids, model_channel_instance_ids)]
        return channel_labels

    def get_channel_labels(self, sem_inst_format='{}_{}'):
        return self._get_channel_labels(self.model_channel_semantic_ids, self.instance_count_id_list,
                                        self.semantic_class_names, map_to_semantic=self.map_to_semantic,
                                        sem_inst_format=sem_inst_format)

    def get_model_channel_labels(self, sem_inst_format='{}_{}'):
        return self._get_channel_labels(self.model_channel_semantic_ids, self.instance_count_id_list,
                                        self.semantic_class_names, map_to_semantic=False,
                                        sem_inst_format=sem_inst_format)

    def decouple_instance_result(self, instance_scores):
        # TODO(allie): implement.
        raise NotImplementedError

    @property
    def channel_values(self):
        return list(range(len(self.model_channel_semantic_ids)))

    def aggregate_across_same_sem_cls(self, arr, empty_val=np.nan):
        """
        Takes a NxC array and converts to an NxS array, where C is the number of channels (multiple instances per object
        class) and S is the number of semantic classes
        """
        if torch.is_tensor(arr):
            raise NotImplementedError
        else:
            assert type(arr) is np.ndarray
        assert arr.shape[1] == self.n_classes
        aggregated_arr = empty_val * np.ones((arr.shape[0], self.n_semantic_classes))
        for sem_idx, sem_val in enumerate(self.semantic_vals):
            channel_idxs = [ci for ci, cname in enumerate(self.model_channel_semantic_vals) if cname == sem_val]
            aggregated_arr[:, sem_idx] = arr[:, channel_idxs].sum(axis=1)
        # assert aggregated_arr.nansum(axis=1) == arr.nansum(axis=1)
        return aggregated_arr

    def decompose_semantic_and_instance_labels(self, gt_combined):
        void_value = self.void_value
        channel_values = self.channel_values
        model_channel_semantic_ids = self.model_channel_semantic_ids
        instance_count_id_list = self.instance_count_id_list
        sem_lbl, inst_lbl = decompose_semantic_and_instance_labels(
            gt_combined,
            channel_inst_vals=channel_values, channel_sem_vals=model_channel_semantic_ids,
            instance_count_id_list=instance_count_id_list, void_value=void_value)
        return sem_lbl, inst_lbl

    def decompose_semantic_and_instance_labels_with_original_sem_ids(self, gt_combined):
        void_value = self.void_value
        channel_values = self.channel_values
        model_channel_semantic_ids = self.model_channel_semantic_vals
        instance_count_id_list = self.instance_count_id_list
        sem_lbl, inst_lbl = decompose_semantic_and_instance_labels(
            gt_combined,
            channel_inst_vals=channel_values, channel_sem_vals=model_channel_semantic_ids,
            instance_count_id_list=instance_count_id_list, void_value=void_value)
        return sem_lbl, inst_lbl

    def create_channel_set_that_fits_all_labels(self, max_n_instances_per_thing=255, sem_type='val'):
        """
        sem_type = 'val' or 'channel_id' based on whether to use the index into the labels table (channel_id) or l[
        'id'] ('val')
        """
        unique_sem_vals = {'val': self.semantic_vals, 'channel_id': self.semantic_ids}[sem_type]
        thing_vals = {'val': self.thing_class_vals, 'channel_id': self.thing_class_ids}[sem_type]
        return create_channel_set_that_fits_all_labels(
            unique_sem_vals=unique_sem_vals, is_thing_each_sem=[True if sv in thing_vals else False
                                                                for sv in unique_sem_vals],
            max_n_instances_per_thing=max_n_instances_per_thing,
            min_inst_id_for_thing=(0 if self.include_instance_channel0 else 1))


def create_channel_set_that_fits_all_labels(unique_sem_vals, is_thing_each_sem, max_n_instances_per_thing,
                                            min_inst_id_for_thing=1):
    channel_sem_vals = []
    channel_inst_vals = []
    for sv, ist in zip(unique_sem_vals, is_thing_each_sem):
        if not ist:
            channel_sem_vals.append(sv)
            channel_sem_vals.append(sv)
        else:
            channel_sem_vals.extend([sv] * max_n_instances_per_thing)
            channel_sem_vals.extend(list(x + min_inst_id_for_thing for x in range(max_n_instances_per_thing)))
    return channel_sem_vals, channel_inst_vals


def decompose_semantic_and_instance_labels(gt_combined, channel_inst_vals, channel_sem_vals,
                                           instance_count_id_list, void_value=-1):
    if torch.is_tensor(gt_combined):
        sem_lbl = gt_combined.clone()
        inst_lbl = gt_combined.clone()
    else:
        sem_lbl = gt_combined.copy()
        inst_lbl = gt_combined.copy()
    sem_lbl[...] = void_value
    inst_lbl[...] = void_value

    for inst_idx, sem_cls, inst_count_id in zip(channel_inst_vals, channel_sem_vals, instance_count_id_list):
        sem_lbl[gt_combined == inst_idx] = sem_cls
        inst_lbl[gt_combined == inst_idx] = inst_count_id
    return sem_lbl, inst_lbl


def label_tuple_to_channel_ids(sem_lbl, inst_lbl, channel_semantic_values, channel_instance_values,
                               set_extras_to_void=True, void_value=-1):
    """
    sem_lbl is size(img); inst_lbl is size(img).  inst_lbl is just the original instance
    image (inst_lbls at coordinates of person 0 are 0)
    """
    assert len(channel_semantic_values) == len(channel_instance_values)
    # TODO(allie): handle class overflow (from ground truth)
    assert set_extras_to_void == True, NotImplementedError
    assert sem_lbl.shape == inst_lbl.shape
    if torch.is_tensor(inst_lbl):
        y = inst_lbl.clone()
    else:
        y = inst_lbl.copy()
    y[...] = void_value
    for sem_inst_idx, (sem_val, inst_val) in enumerate(zip(channel_semantic_values, channel_instance_values)):
        y[(sem_lbl == int(sem_val)) * (inst_lbl == int(inst_val))] = sem_inst_idx
    # potential bug: y produces void values where they shouldn't exist
    if torch.is_tensor(y):
        if torch.any((y == void_value).int() - (inst_lbl == void_value).int()):
            raise Exception
    else:
        if np.any(inst_lbl[(y == void_value) != void_value]):
            semantic_vals = np.unique(sem_lbl[y == void_value])
            inst_vals = {}
            for sem_val in semantic_vals:
                inst_vals[sem_val] = np.unique(inst_lbl[(y == void_value) * (sem_lbl == sem_val)])
            print(inst_vals)
            raise Exception
    # if (y == void_value)
    return y


def get_instance_to_semantic_mapping_from_model_channel_semantic_ids(model_channel_semantic_ids,
                                                                     as_numpy=False, compose_transposed=True):
    """
    returns a binary matrix, where semantic_instance_mapping is N x S
    (N = # instances, S = # semantic classes)
    semantic_instance_mapping[inst_idx, :] is a one-hot vector,
    and semantic_instance_mapping[inst_idx, sem_idx] = 1 iff that instance idx is an instance
    of that semantic class.
    compose_transposed: S x N
    """
    n_instance_classes = len(model_channel_semantic_ids)
    n_semantic_classes = int(max(model_channel_semantic_ids) + 1)
    if not compose_transposed:
        instance_to_semantic_mapping_matrix = torch.zeros((n_instance_classes, n_semantic_classes)).float()
        for instance_idx, semantic_idx in enumerate(model_channel_semantic_ids):
            instance_to_semantic_mapping_matrix[instance_idx, semantic_idx] = 1
    else:
        instance_to_semantic_mapping_matrix = torch.zeros((n_semantic_classes, n_instance_classes)).float()
        for instance_idx, semantic_idx in enumerate(model_channel_semantic_ids):
            instance_to_semantic_mapping_matrix[semantic_idx, instance_idx] = 1
    return instance_to_semantic_mapping_matrix if not as_numpy else \
        instance_to_semantic_mapping_matrix.numpy()


def permute_labels(label_preds, permutations):
    if torch.is_tensor(label_preds):
        label_preds_permuted = label_preds.clone()
    else:
        label_preds_permuted = label_preds.copy()
    for idx in range(permutations.shape[0]):
        permutation = permutations[idx, :]
        for new_channel, old_channel in enumerate(permutation):
            label_preds_permuted[label_preds == old_channel] = new_channel
    return label_preds_permuted


def create_default_labels_table_from_instance_problem_config(semantic_class_names, semantic_class_vals, colors=None,
                                                             supercategories=None, isthing=None):
    if supercategories is None:
        supercategories = semantic_class_names
    if isthing is None:
        isthing = [1 if name != 'background' else 0 for name in semantic_class_names]
    labels_table = [
        CategoryCOCOFormat(
            **{'id': semantic_class_vals[i],
               'name': name,
               'color': colors[i],
               'supercategory': supercategories[i],
               'isthing': isthing[i]})
        for i, name in enumerate(semantic_class_names)]
    return labels_table
