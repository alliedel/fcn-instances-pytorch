#!/usr/bin/env python

from . import voc_raw
import numpy as np
from torch.utils import data

# TODO(allie): Allow for permuting the instance order at the beginning, and copying each filename
#  multiple times with the assigned permutation.  That way you can train in batches that have
# different permutations for the same image (may affect training if batched that way).
# You may also want to permute different semantic classes differently, though I'm pretty sure
# the network shouldn't be able to understand that's going on (semantic classes are handled
# separately)

DEBUG_ASSERT = True


class VOCMappedToInstances(voc_raw.VOCWithTransformations):
    def __init__(self, root, split='train', void_value=-1, resize=False, resize_size=None,
                 cutoff_instances_above=None, make_all_instance_classes_semantic=False):
        """
        cutoff_instances_above: int, max number of instances.
        Note inst_lbl == 0 indicates non-instance label ('rest of them').  We can handle this
        accordingly after we load the labels.
        """

        super(VOCMappedToInstances, self).__init__(root, split=split, resize=resize,
                                                   resize_size=resize_size)
        self.void_value = void_value
        self.background_value = 0
        self.cutoff_instances_above = cutoff_instances_above
        self.make_all_instance_classes_semantic = make_all_instance_classes_semantic

        # Get dictionary of raw id assignments (semantic, instance, void, background)
        self._raw_id_list, self._raw_id_assignments = get_raw_id_assignments()
        # Override if asked to
        if self.make_all_instance_classes_semantic:
            self._raw_id_assignments['instance'] = [False for _ in
                                                    self._raw_id_assignments['instance']]

        # Set list of corresponding training ids, and get training id assignments
        self._raw_id_to_train_id, self.train_id_list, self.train_id_assignments = \
            get_train_id_assignments(self._raw_id_assignments, void_value, background_value=0)

        # Get dictionary of class names for each type (semantic, void, background, instance)
        self.class_names = []
        for train_id in self.train_id_list:
            raw_ids_mapped_to_this = [i for i, _ in enumerate(self._raw_id_list)
                                      if self._raw_id_to_train_id[i] == train_id]
            class_name = ','.join([voc_raw.class_names[i] for i in
                                   raw_ids_mapped_to_this])
            self.class_names.append(class_name)

    def __getitem__(self, index):
        # TODO(allie): Possibly change this convention:
        """
        Returns sem_lbl, inst_lbl where:
        inst_lbl >= 1 for instance classes if instances have been identified
                    --> we remap these to inst_lbl - 1, so they start from index 0.
        inst_lbl == 0 for semantic classes
        inst_lbl == -1 for instance classes that have not been identified
        Note for instance classes, the 'extras' are -1, but for semantic classes,
            the 'extras' (all stuff) are 0.  This ensures we can ignore the instance leftovers
            while still penalizing the semantic classes, without having to carry around
            identifiers of whether a class is semantic or instance.
        Note -- when the number of instances for an instance class is just 1, we will
        break convention and convert it to a semantic class instead.
        """
        img, (sem_lbl, inst_lbl) = super(VOCMappedToInstances, self).__getitem__(index)
        sem_lbl = map_raw_sem_ids_to_train_ids(sem_lbl, self._raw_id_list, self._raw_id_to_train_id)
        inst_lbl = self.raw_inst_to_train_inst_labels(inst_lbl, sem_lbl)
        return img, (sem_lbl, inst_lbl)

    def modify_length(self, modified_length):
        self.files[self.split] = self.files[self.split][:modified_length]

    def copy(self, modified_length=10):
        my_copy = VOCMappedToInstances(root=self.root)
        for attr, val in self.__dict__.items():
            setattr(my_copy, attr, val)
        assert modified_length <= len(my_copy), "Can\'t create a copy with more examples than " \
                                                "the initial dataset"
        self.modify_length(modified_length)
        return my_copy

    def raw_inst_to_train_inst_labels(self, inst_lbl, sem_lbl):
        inst_lbl = map_raw_inst_labels_to_instance_count(inst_lbl)
        for (sem_train_id, is_instance, is_semantic) in \
                zip(self.train_id_list, self.train_id_assignments['instance'],
                    self.train_id_assignments['semantic']):
            if not is_instance and is_semantic:
                inst_lbl[sem_lbl == sem_train_id] = 0
            elif not is_semantic:
                inst_lbl[sem_lbl == sem_train_id] = -1
            else:
                # all instance labels at this point should be in the range (1, n_instances + 1)
                # remap to (0, n_instances).
                # Note instance labels given the label 0 will now be void (we don't use
                # the loss from the leftovers in this implementation, though that will likely
                #  have to change)
                # inst_lbl[sem_lbl == sem_train_id] -= 1
                pass
        if self.cutoff_instances_above is not None:
            # Assign 0 to any instances that are not in the range (1, cutoff + 1) -- leftovers
            inst_lbl[inst_lbl > self.cutoff_instances_above] = 0

        return inst_lbl


def get_train_id_assignments(raw_id_assignments, void_value, background_value=0):
    # train ids include void_value, background_value, and range(1, n_semantic_classes + 1)
    n_raw_classes = len(raw_id_assignments['semantic'])
    n_semantic_classes = raw_id_assignments['semantic'].count(True)
    assert background_value == 0, NotImplementedError
    # Each semantic class get its own value
    train_ids = [void_value, background_value] + list(range(1, n_semantic_classes))

    # Map raw ids to unique train ids
    sem_cls = 0
    raw_id_to_train_id = []
    for raw_id in range(n_raw_classes):
        if raw_id_assignments['background'][raw_id]:
            raw_id_to_train_id.append(background_value)
        elif raw_id_assignments['semantic'][raw_id]:
            sem_cls += 1
            raw_id_to_train_id.append(sem_cls)
        elif raw_id_assignments['void'][raw_id]:
            raw_id_to_train_id.append(void_value)
        else:
            raise Exception('raw_id_assignments does not cover all the classes')
    assert sem_cls == n_semantic_classes - sum(raw_id_assignments['background']), \
        'Debug assert failed here.'

    is_background, is_void, is_semantic, is_instance = [], [], [], []
    for tid in train_ids:
        # background
        is_background.append(tid == background_value)
        # void
        is_void.append(tid == void_value)
        my_raw_ids = [ci for ci, mapped_train_id in enumerate(raw_id_to_train_id)
                      if mapped_train_id == tid]
        # semantic
        is_semantic_for_all_raw = [raw_id_assignments['semantic'][id] for id in my_raw_ids]
        if not all(is_semantic_for_all_raw[0] == is_sem for is_sem in is_semantic_for_all_raw):
            raise Exception('Mapped semantic, non-semantic raw classes to the same train id.')
        is_semantic.append(is_semantic_for_all_raw[0])
        # instance
        is_instance_for_all_raw = [raw_id_assignments['instance'][id] for id in my_raw_ids]
        if not all(is_instance_for_all_raw[0] == is_inst for is_inst in is_instance_for_all_raw):
            raise Exception('Mapped instance, non-instance raw classes to the same train id')
        is_instance.append(is_instance_for_all_raw[0])

    train_assignments = {'background': is_background,
                         'void': is_void,
                         'semantic': is_semantic,
                         'instance': is_instance}
    return raw_id_to_train_id, train_ids, train_assignments


def get_raw_id_assignments():
    """
    # semantic: All classes we are considering (non-void)
    # instance: Subset of semantic classes with instances
    # void: Any classes that don't get evaluated on (too few, or they're unlabeled, etc.)
    # background: Any classes that get mapped into one big 'miscellanious' class (differs from
      void because we evaluate on it)
    """
    raw_id_list = voc_raw.ids
    is_background = [name == 'background'
                     for c, name in enumerate(voc_raw.class_names)]
    is_semantic = [not is_void or is_background[ci] for ci, is_void in enumerate(
        voc_raw.is_void)]
    has_instances = [has_instances and not is_void for ci, (has_instances, is_void)
                     in enumerate(zip(voc_raw.has_instances,
                                      voc_raw.is_void))]
    is_void = [is_void and not is_background[ci] for ci, is_void in enumerate(
        voc_raw.is_void)]
    assert len(is_void) == len(is_background) == len(is_semantic) == len(has_instances)
    assert all([(b and s) == b for b, s in zip(is_background, is_semantic)])
    return raw_id_list, {'semantic': is_semantic,
                         'instance': has_instances,
                         'void': is_void,
                         'background': is_background}


def map_raw_inst_labels_to_instance_count(inst_lbl):
    return inst_lbl


def map_raw_sem_ids_to_train_ids(sem_lbl, old_values, new_values_from_old_values):
    """
    Specifically for Cityscapes. There are a bunch of classes that didn't get used,
    so they 'remap' them onto actual training classes.  Leads to very silly remapping after
    loading...

    WARNING: We map iteratively (less memory-intensive than copying sem_lbl or making masks).
    For this to work, train_ids must be <= ids, background_value <= background_ids,
    void_value <= void_ids.
    """
    new_value_sorted_idxs = np.argsort(new_values_from_old_values)
    old_values = [old_values[i] for i in new_value_sorted_idxs]
    new_values_from_old_values = [new_values_from_old_values[i] for i in new_value_sorted_idxs]
    assert len(old_values) == len(new_values_from_old_values)
    assert all([new <= old for old, new in zip(old_values, new_values_from_old_values)]), \
        NotImplementedError('I\'ve got to do something smarter when assigning...')
    # map ids to train_ids (e.g. - tunnel (id=16) is unused, so maps to 255.
    for old_val, new_val in zip(old_values, new_values_from_old_values):
        sem_lbl[sem_lbl == old_val] = new_val
    return sem_lbl
