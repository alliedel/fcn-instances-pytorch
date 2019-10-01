import instanceseg.utils.imgutils
from instanceseg.datasets import coco_format
from instanceseg.utils.torch_utils import fast_remap
from . import labels_table_cityscapes
from .precomputed_file_transformations import PrecomputedDatasetFileTransformerBase
import numpy as np
import os.path as osp
import os
import PIL.Image
from instanceseg.utils import misc, datasets

NON_VOID_BACKGROUND_CLASS_NAMES = ('ego vehicle',)


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
        datasets.write_np_array_as_img_with_colormap_palette(arr, new_file, palette)
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
    new_sem_file_tag = 'CustomTrainIdsWithBground0'
    old_inst_file_tag = 'Ids'
    new_inst_file_tag = 'CustomTrainIdsWithBground0'
    void_value = 255
    background_value = 0
    background_color = (255, 255, 255)

    def __init__(self):
        self.standard_id_to_train_id = labels_table_cityscapes.ID_TO_TRAIN_ID
        self.standard_train_ids_labels_table = labels_table_cityscapes.get_cityscapes_trainids_label_table_cocoform()
        self.original_semantic_class_names = [l['name'] for l in self.original_labels_table]
        self.labels_table = self.get_labels_table()
        self.id_to_train_id = {rawid: self.new_train_id_from_raw(rawid)
                               for rawid in sorted([l['id'] for l in self.original_labels_table])}

    def get_labels_table(self):
        assert self.background_color not in [l.color for l in self.standard_train_ids_labels_table]  # we're going to
        # need this
        labels_table = [coco_format.CategoryCOCOFormat(id=0, name='background', color=self.background_color,
                                                       supercategory='background', isthing=False)]
        for l in self.standard_train_ids_labels_table:
            if l.id == -1:  # not adding license plate for now...
                continue
            new_train_id = self.new_train_id_from_old_train_id(l.id)
            labels_table.append(coco_format.CategoryCOCOFormat(id=new_train_id, name=l.name, color=l.color,
                                                               supercategory=l.supercategory, isthing=l.isthing))
        return labels_table

    def new_train_id_from_raw(self, raw_val):
        old_train_id = labels_table_cityscapes.ID_TO_TRAIN_ID[raw_val]
        return self.new_train_id_from_old_train_id(old_train_id)

    def new_train_id_from_old_train_id(self, old_train_id):
        assert old_train_id in [l.id for l in self.standard_train_ids_labels_table], '{} not in {}'.format(
            old_train_id, [l.id for l in self.standard_train_ids_labels_table])
        return old_train_id if old_train_id in (-1, 255) else old_train_id + 1

    @property
    def original_labels_table(self):
        return labels_table_cityscapes.CITYSCAPES_LABELS_TABLE

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
        Maps all instance labels to either 0 if semantic or [1, ..., n_instances]
        """
        inst_lbl = map_raw_inst_labels_to_instance_count(inst_lbl, sem_lbl)
        for l in self.labels_table:
            if l.id == self.void_value:
                inst_lbl[sem_lbl == l.id] = self.void_value
            elif l.id < 0:
                raise Exception('We shouldnt have any semantic value < 0 in here (license plate?)')
            elif l.isthing:
                pass
            else: # not l.isthing:  # shouldnt be instances
                assert (inst_lbl[sem_lbl == l.id]).sum() == 0

        return inst_lbl

    def generate_train_id_instance_file(self, raw_format_inst_lbl_file, new_format_inst_lbl_file, sem_lbl_file):
        sem_lbl = instanceseg.utils.imgutils.load_img_as_dtype(sem_lbl_file, np.int32)
        inst_lbl = instanceseg.utils.imgutils.load_img_as_dtype(raw_format_inst_lbl_file, np.int32)
        inst_lbl = self.raw_inst_to_train_inst_labels(inst_lbl, sem_lbl)
        orig_lbl = PIL.Image.open(raw_format_inst_lbl_file)
        if orig_lbl.mode == 'P':
            datasets.write_np_array_as_img_with_borrowed_colormap_palette(
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
        sem_lbl = instanceseg.utils.imgutils.load_img_as_dtype(raw_id_sem_lbl_file, np.int32)
        sem_lbl = self.map_raw_sem_ids_to_train_ids(sem_lbl)
        datasets.write_np_array_as_img_with_borrowed_colormap_palette(
            sem_lbl, new_train_id_sem_lbl_file, filename_for_colormap=raw_id_sem_lbl_file)

    @property
    def semantic_class_names(self):
        return [l.name for l in self.labels_table]

    def transform_semantic_class_names(self, original_semantic_class_names):
        # Only works with one set of original semantic classes
        assert original_semantic_class_names == self.original_semantic_class_names

        return self.semantic_class_names

    def untransform_semantic_class_names(self):
        return self.original_semantic_class_names

    def transform_labels_table(self, original_labels_table=None):
        assert original_labels_table is None or all(l1 == l2 for l1, l2 in zip(original_labels_table,
                                                                               self.original_labels_table))
        return self.get_labels_table()

    def map_raw_sem_ids_to_train_ids(self, sem_lbl):
        """
        Specifically for Cityscapes. There are a bunch of classes that didn't get used,
        so they 'remap' them onto actual training classes.  Leads to very silly remapping after
        loading...

        WARNING: We map iteratively (less memory-intensive than copying sem_lbl or making masks).
        For this to work, train_ids must be <= ids, background_value <= background_ids,
        void_value <= void_ids.
        """
        fast_remap(sem_lbl, list(self.id_to_train_id.keys()), list(self.id_to_train_id.values()))
        return sem_lbl


def map_raw_inst_labels_to_instance_count(inst_lbl, sem_lbl_for_verification=None):
    """
    Specifically for Cityscapes.
    Warning: inst_lbl must be an int/long for this to work
    """
    orig_inst_lbl = inst_lbl.copy()
    inst_lbl[inst_lbl < 1000] = 0
    sem_lbl_of_objects = np.int32(inst_lbl / 1000)
    for sem_val in np.unique(sem_lbl_of_objects):
        if sem_val == 0:
            continue
        unique_inst_ids = sorted(np.unique(orig_inst_lbl[sem_lbl_of_objects == sem_val]))
        if max(unique_inst_ids) != sem_val * 1000 + len(unique_inst_ids) - 1:
            new_consecutive_inst_ids = range(1000 * sem_val, 1000 * sem_val + len(unique_inst_ids))
            print(misc.color_text('Instance values were in a weird format! Values present: {}.  Missing: {}'.format(
                unique_inst_ids, set(new_consecutive_inst_ids) - set(unique_inst_ids)),
                misc.TermColors.WARNING))
            fast_remap(inst_lbl, old_vals=unique_inst_ids, new_vals=new_consecutive_inst_ids)

    inst_lbl -= np.int32(sem_lbl_of_objects) * np.int32(1000)  # more efficient
    inst_lbl[orig_inst_lbl >= 1000] += 1

    if sem_lbl_for_verification is not None:
        try:
            sem_lbl_reconstructed = sem_lbl_of_objects
            assert np.all((sem_lbl_reconstructed == 0) == (orig_inst_lbl < 1000))
            sem_lbl_reconstructed[orig_inst_lbl < 1000] = orig_inst_lbl[orig_inst_lbl < 1000]
            assert np.all(sem_lbl_reconstructed == sem_lbl_for_verification)
        except AssertionError:
            import ipdb; ipdb.set_trace()
            raise

    return inst_lbl


