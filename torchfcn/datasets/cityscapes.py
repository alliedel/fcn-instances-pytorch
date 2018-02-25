from torch.utils import data
import labels_table_cityscapes
from torchfcn.datasets import cityscapes_raw


class CityscapesMappedToInstances(data.Dataset):
    def __init__(self, root, split='train',
                 void_value=-1, resize=True, resize_size=(512, 1024)):
        self.raw_dataset = cityscapes_raw.CityscapesWithTransformations(
            root, split=split, resize=resize, resize_size=resize_size)
        self.void_value = void_value
        self.background_value = 0

        # Get dictionary of raw id assignments (semantic, instance, void, background)
        self._raw_id_list, self._raw_id_assignments = get_raw_id_assignments()

        # Set list of corresponding training ids, and get training id assignments
        self._raw_id_to_train_id, self.train_id_list, self.train_id_assignments = \
            get_train_id_assignments(self._raw_id_assignments, void_value, background_value=0)

        # Get dictionary of class names for each type (semantic, void, background, instance)
        self.class_names = []
        for train_id in self.train_id_list:
            raw_ids_mapped_to_this = [raw_id for raw_id in self._raw_id_list
                                      if self._raw_id_assignments[raw_id] == train_id]
            class_name = ','.join(labels_table_cityscapes.class_names[raw_ids_mapped_to_this])
            self.class_names.append(class_name)

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, index):
        img, (sem_lbl, inst_lbl) = self.raw_dataset[index]

        sem_lbl = map_raw_sem_ids_to_train_ids(sem_lbl, self._raw_id_list, self._raw_id_to_train_id)
        inst_lbl = map_raw_inst_labels_to_instance_count(inst_lbl)
        return img, (sem_lbl, inst_lbl)

    def modify_length(self, modified_length):
        self.raw_dataset.files[self.raw_dataset.split] = self.raw_dataset.files[
                                                             self.raw_dataset.split][
                                                         :modified_length]

    def copy(self, modified_length=10):
        my_copy = CityscapesMappedToInstances(root=self.raw_dataset.root)
        for attr, val in self.__dict__.items():
            setattr(my_copy, attr, val)
        assert modified_length <= len(my_copy), "Can\'t create a copy with more examples than " \
                                                "the initial dataset"
        self.modify_length(modified_length)
        return my_copy


def get_train_id_assignments(raw_id_assignments, void_value, background_value=0):
    # train ids include void_value, background_value, and range(1, n_semantic_classes + 1)
    n_raw_classes = len(raw_id_assignments['semantic'])
    n_semantic_classes = raw_id_assignments['semantic'].count(True)
    semantic_values = range(n_semantic_classes + 1)
    assert void_value not in semantic_values, 'void_value should not overlap with semantic classes'
    assert background_value == 0, NotImplementedError
    # Each semantic class get its own value
    train_ids = [void_value, background_value].append(range(n_semantic_classes))
    import ipdb;
    ipdb.set_trace()
    train_assignments = {'background': [tid == background_value for tid in train_ids],
                         'void': [tid == void_value for tid in train_ids],
                         'semantic': [tid not in semantic_values for tid in train_ids]}
    # Map raw ids to unique train ids
    sem_cls = 0
    raw_id_to_train_id = []
    for raw_id in range(n_raw_classes):
        if raw_id_assignments['semantic']:
            sem_cls += 1
            raw_id_to_train_id.append(sem_cls)
        elif raw_id_assignments['void']:
            raw_id_to_train_id.append(void_value)
        elif raw_id_assignments['background']:
            raw_id_to_train_id.append(background_value)
        else:
            raise Exception('raw_id_assignments does not cover all the classes')
    assert sem_cls == n_semantic_classes, 'Debug assert failed here.'
    return raw_id_to_train_id, train_ids, train_assignments


def get_raw_id_assignments():
    """
    # semantic: All classes we are considering (non-void)
    # instance: Subset of semantic classes with instances
    # void: Any classes that don't get evaluated on (too few, or they're unlabeled, etc.)
    # background: Any classes that get mapped into one big 'miscellanious' class (differs from
      void because we evaluate on it)
    """
    raw_id_list = labels_table_cityscapes.ids
    is_semantic = [not is_void for ci, is_void in enumerate(labels_table_cityscapes.is_void)]
    has_instances = [has_instances for ci, has_instances
                     in enumerate(labels_table_cityscapes.has_instances)]
    is_void = [is_void for ci, is_void in enumerate(labels_table_cityscapes.is_void)
               if is_void]
    is_background = [name == 'unlabeled'
                     for c, name in enumerate(labels_table_cityscapes.class_names)]
    return raw_id_list, {'semantic': is_semantic,
                         'instance': has_instances,
                         'void': is_void,
                         'background': is_background}


def map_raw_inst_labels_to_instance_count(inst_lbl):
    """
    Specifically for Cityscapes.
    Warning: inst_lbl must be an int/long for this to work
    """
    inst_lbl -= (inst_lbl / 1000) * 1000  # more efficient mod(inst_lbl, 1000)
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
    assert len(old_values) == len(new_values_from_old_values)
    assert all([old < new for old, new in zip(old_values, new_values_from_old_values)]), \
        NotImplementedError('I\'ve got to do something smarter when assigning...')
    # map ids to train_ids (e.g. - tunnel (id=16) is unused, so maps to 255.
    for old_val, new_val in zip(old_values, new_values_from_old_values):
        sem_lbl[sem_lbl == old_val] = new_val
    return sem_lbl
