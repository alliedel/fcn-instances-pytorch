from . import labels_table_cityscapes
from .precomputed_file_transformations import PrecomputedDatasetFileTransformerBase
import numpy as np
import os.path as osp
import os
from . import dataset_utils
import PIL.Image
from torchfcn.utils import misc


def convert_to_p_mode_file(old_file, new_file, palette, assert_inside_palette_range=True):
    im = PIL.Image.open(old_file)
    if assert_inside_palette_range:
        max_palette_val = int(len(palette.getpalette()) / 3 - 1)
        min_val, max_val = im.getextrema()
        assert min_val >= 0
        assert max_val <= max_palette_val, '{}:\nmax value, {}, > palette max, {}'.format(old_file, max_val,
                                                                                          max_palette_val)

    if im.mode == 'P':  # already mode p.  symlink so we dont go through this again.
        os.symlink(old_file, new_file)
    elif im.mode == 'I':
        arr = np.array(im)
        dataset_utils.write_np_array_as_img_with_colormap_palette(arr, new_file, palette)
    else:  # if im.mode == 'RGB':
        converted = im.quantize(palette=palette)
        converted.save(new_file)


class ConvertLblstoPModePILImages(PrecomputedDatasetFileTransformerBase):
    old_sem_file_tag = '.png'
    new_sem_file_tag = '_mode_p.png'
    old_inst_file_tag = '.png'
    new_inst_file_tag = '_mode_p.png'

    def __init__(self):
        self.semantic_palette = \
            labels_table_cityscapes.get_semantic_palette_image()
        self.instance_palette = \
            labels_table_cityscapes.get_instance_palette_image()

    def transform(self, img_file, sem_lbl_file, inst_lbl_file):
        assert self.old_sem_file_tag in sem_lbl_file and self.old_sem_file_tag in inst_lbl_file
        new_sem_lbl_file = sem_lbl_file.replace(self.old_sem_file_tag, self.new_sem_file_tag)
        if not osp.isfile(new_sem_lbl_file):
            assert osp.isfile(sem_lbl_file), '{} does not exist'.format(sem_lbl_file)
            print('Creating {} from {}'.format(new_sem_lbl_file, sem_lbl_file))
            convert_to_p_mode_file(sem_lbl_file, new_sem_lbl_file, palette=self.semantic_palette,
                                   assert_inside_palette_range=True)
        new_inst_lbl_file = inst_lbl_file.replace(self.old_inst_file_tag, self.new_inst_file_tag)
        if not osp.isfile(new_inst_lbl_file):
            assert osp.isfile(inst_lbl_file), '{} does not exist'.format(inst_lbl_file)
            print('Creating {} from {}'.format(new_inst_lbl_file, inst_lbl_file))
            convert_to_p_mode_file(inst_lbl_file, new_inst_lbl_file, palette=self.instance_palette,
                                   assert_inside_palette_range=True)
        return img_file, new_sem_lbl_file, inst_lbl_file

    def untransform(self, img_file, sem_lbl_file, inst_lbl_file):
        old_sem_lbl_file = sem_lbl_file.replace(self.new_sem_file_tag, self.old_sem_file_tag)
        old_inst_lbl_file = inst_lbl_file.replace(self.new_inst_file_tag, self.old_inst_file_tag)
        assert osp.isfile(old_sem_lbl_file)
        return img_file, old_sem_lbl_file, old_inst_lbl_file


class CityscapesMapRawtoTrainIdPrecomputedFileDatasetTransformer(PrecomputedDatasetFileTransformerBase):
    old_sem_file_tag = 'Ids'
    new_sem_file_tag = 'TrainIds'
    old_inst_file_tag = 'Ids'
    new_inst_file_tag = 'TrainIds'
    void_value = 255
    background_value = 0

    def __init__(self):
        # Get dictionary of raw id assignments (semantic, instance, void, background)
        self._raw_id_list, self._raw_id_assignments = get_raw_id_assignments()

        # Set list of corresponding training ids, and get training id assignments
        self._raw_id_to_train_id, self.train_id_list, self.train_id_assignments = \
            get_train_id_assignments(self._raw_id_assignments, void_value=self.void_value,
                                     background_value=self.background_value)
        self.original_semantic_class_names = None

    def transform(self, img_file, sem_lbl_file, inst_lbl_file):
        new_sem_lbl_file = sem_lbl_file.replace(self.old_sem_file_tag, self.new_sem_file_tag)
        if not osp.isfile(new_sem_lbl_file):
            print('Generating {}'.format(new_sem_lbl_file))
            assert osp.isfile(sem_lbl_file), '{} does not exist'.format(sem_lbl_file)
            self.generate_train_id_semantic_file(sem_lbl_file, new_sem_lbl_file)
        new_inst_lbl_file = inst_lbl_file.replace(self.old_inst_file_tag, self.new_inst_file_tag)
        if not osp.isfile(new_inst_lbl_file):
            print('Generating {}'.format(new_inst_lbl_file))
            assert osp.isfile(inst_lbl_file), '{} does not exist'.format(inst_lbl_file)
            self.generate_train_id_instance_file(inst_lbl_file, new_inst_lbl_file, sem_lbl_file)
        if 1:
            assert osp.isfile(new_inst_lbl_file)
            assert osp.isfile(new_sem_lbl_file)

        return img_file, new_sem_lbl_file, new_inst_lbl_file

    def untransform(self, img_file, sem_lbl_file, inst_lbl_file):
        old_sem_lbl_file = sem_lbl_file.replace(self.new_sem_file_tag, self.old_sem_file_tag)
        old_inst_lbl_file = sem_lbl_file.replace(self.new_inst_file_tag, self.old_inst_file_tag)
        assert osp.isfile(old_sem_lbl_file)
        return img_file, old_sem_lbl_file, old_inst_lbl_file

    def raw_inst_to_train_inst_labels(self, inst_lbl, sem_lbl):
        """
        Maps all instance labels to either 0 if semantic or [1, ..., n_instances], (-1 <-- if instance id >
        n_instances)
        """
        inst_lbl = map_raw_inst_labels_to_instance_count(inst_lbl)
        for (sem_train_id, is_instance, is_semantic) in \
                zip(self.train_id_list, self.train_id_assignments['instance'],
                    self.train_id_assignments['semantic']):
            if not is_instance and is_semantic:
                inst_lbl[sem_lbl == sem_train_id] = 0
            elif not is_semantic:
                inst_lbl[sem_lbl == sem_train_id] = -1
            else:
                pass
        return inst_lbl

    def generate_train_id_instance_file(self, raw_format_inst_lbl_file, new_format_inst_lbl_file, sem_lbl_file):
        sem_lbl = dataset_utils.load_img_as_dtype(sem_lbl_file, np.int32)
        inst_lbl = dataset_utils.load_img_as_dtype(raw_format_inst_lbl_file, np.int32)
        inst_lbl = self.raw_inst_to_train_inst_labels(inst_lbl, sem_lbl)
        orig_lbl = PIL.Image.open(raw_format_inst_lbl_file)
        if orig_lbl.mode == 'P':
            dataset_utils.write_np_array_as_img_with_borrowed_colormap_palette(
                inst_lbl, new_format_inst_lbl_file, filename_for_colormap=raw_format_inst_lbl_file)
        elif orig_lbl.mode == 'I':
            new_img_data = PIL.Image.fromarray(inst_lbl, mode='I')
            new_lbl_img = orig_lbl.copy()
            new_lbl_img.paste(new_img_data)
            new_lbl_img.save(new_format_inst_lbl_file)
        else:
            raise NotImplementedError

    def generate_train_id_semantic_file(self, raw_id_sem_lbl_file, new_train_id_sem_lbl_file):
        print('Generating per-semantic instance file: {}'.format(new_train_id_sem_lbl_file))
        sem_lbl = dataset_utils.load_img_as_dtype(raw_id_sem_lbl_file, np.int32)
        sem_lbl = map_raw_sem_ids_to_train_ids(sem_lbl, old_values=self._raw_id_list,
                                               new_values_from_old_values=self._raw_id_to_train_id)
        dataset_utils.write_np_array_as_img_with_borrowed_colormap_palette(
            sem_lbl, new_train_id_sem_lbl_file, filename_for_colormap=raw_id_sem_lbl_file)

    def transform_semantic_class_names(self, original_semantic_class_names):
        assert len(original_semantic_class_names) == len(self._raw_id_list)
        # preserve for later
        self.original_semantic_class_names = original_semantic_class_names

        semantic_class_names = []
        # import ipdb; ipdb.set_trace()
        train_ids_for_semantic_list = [train_id for train_id in self.train_id_list if train_id != self.void_value]
        assert all([a == b for a, b in zip(train_ids_for_semantic_list, range(len(train_ids_for_semantic_list)))]), \
            AssertionError('We need the semantic class name indices to match their values.  Bug somewhere.')
        for train_id in train_ids_for_semantic_list:
            if train_id == self.void_value:
                continue
            raw_ids_mapped_to_this = [raw_id for raw_id in self._raw_id_list
                                      if self._raw_id_to_train_id[raw_id] == train_id]
            class_name = ','.join([labels_table_cityscapes.class_names[i] for i in
                                   raw_ids_mapped_to_this])
            if train_id == self.background_value:
                class_name = 'background'
            semantic_class_names.append(class_name)
        # assert len(semantic_class_names) == 19, 'I expected raw Cityscapes to have 19 classes'
        return semantic_class_names

    def untransform_semantic_class_names(self):
        return self.original_semantic_class_names


def get_train_id_assignments(raw_id_assignments, void_value, background_value=0):
    # train ids include void_value, background_value, and range(1, n_semantic_classes + 1)
    n_raw_classes = len(raw_id_assignments['semantic'])
    n_semantic_classes = raw_id_assignments['semantic'].count(True)
    non_bground_semantic_values = range(1, n_semantic_classes + 1)
    assert void_value not in non_bground_semantic_values, 'void_value should not overlap with ' \
                                                          'semantic classes'
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
    raw_id_list = labels_table_cityscapes.ids
    is_background = [name == 'unlabeled' or name == 'background'
                     for c, name in enumerate(labels_table_cityscapes.class_names)]
    is_semantic = [not is_void or is_background[ci] for ci, is_void in enumerate(
        labels_table_cityscapes.is_void)]
    has_instances = [has_instances and not is_void for ci, (has_instances, is_void)
                     in enumerate(zip(labels_table_cityscapes.has_instances,
                                      labels_table_cityscapes.is_void))]
    is_void = [is_void and not is_background[ci] for ci, is_void in enumerate(
        labels_table_cityscapes.is_void)]
    assert len(is_void) == len(is_background) == len(is_semantic) == len(has_instances)
    assert all([(b and s) == b for b, s in zip(is_background, is_semantic)])
    return raw_id_list, {'semantic': is_semantic,
                         'instance': has_instances,
                         'void': is_void,
                         'background': is_background}


def map_raw_inst_labels_to_instance_count(inst_lbl):
    """
    Specifically for Cityscapes.
    Warning: inst_lbl must be an int/long for this to work
    """
    inst_lbl[inst_lbl < 1000] = 0
    inst_lbl[inst_lbl >= 1000] += 1  # We make instance labels start from 1 instead of 0.
    inst_lbl -= np.int32((inst_lbl / 1000)) * np.int32(1000)  # more efficient
    max_lbl = inst_lbl.max()
    if max_lbl > 0:
        # Check if instance values are consecutive, starting from 1.  If not, shift them all.
        consecutive_instance_values = list(range(1, max_lbl + 1))
        is_present = [bool((inst_lbl == val).sum() > 0) for val in consecutive_instance_values]

        # Shift if needed
        if not all(is_present):
            old_instance_values = [val for val, p in zip(consecutive_instance_values, is_present) if p]
            num_present = sum([1 for p in is_present if p])
            new_instance_values = list(range(1, num_present + 1))
            assert num_present < len(consecutive_instance_values), AssertionError('debug')
            assert num_present == len(new_instance_values) == len(old_instance_values), AssertionError('debug')
            print(misc.color_text('Instance values were in a weird format! Values present: {}'.format(
                old_instance_values), misc.TermColors.WARNING))
            for old_val, new_val in zip(old_instance_values, new_instance_values):
                inst_lbl[inst_lbl == old_val] = new_val

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
    if all([new <= old for old, new in zip(old_values, new_values_from_old_values)]):
        # map ids to train_ids (e.g. - tunnel (id=16) is unused, so maps to 255.
        for old_val, new_val in zip(old_values, new_values_from_old_values):
            sem_lbl[sem_lbl == old_val] = new_val
    else:  # less efficient method
        sem_lbl_old = sem_lbl.copy()
        for old_val, new_val in zip(old_values, new_values_from_old_values):
            sem_lbl[sem_lbl_old == old_val] = new_val
    return sem_lbl
