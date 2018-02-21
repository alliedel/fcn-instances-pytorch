#!/usr/bin/env python

import collections
import os.path as osp
import PIL.Image

from . import dataset_utils
from torch.utils import data
import os
import torch
import numpy as np
import labels_table_cityscapes
import local_pyutils

DEBUG_ASSERT = True

# TODO(allie): Allow for augmentations
# TODO(allie): Avoid so much code copying by making a semantic->instance wrapper class or base class
# TODO(allie): Allow for semantic instead of instance segmentation on the semantic classes (
# instead of mapping them all collectively onto background).


logger = local_pyutils.get_logger()

# TODO(allie): Allow shuffling within the dataset here (instead of with train_loader) so we can
# set the loader shuffle to False (so it goes in the same order every time), but still get images
#  out in a random order.
class CityscapesClassSegBase(data.Dataset):
    mean_bgr = np.array([73.15835921, 82.90891754, 72.39239876])

    def __init__(self, root, split='train', transform=False, n_max_per_class=1,
                 semantic_subset=None, map_other_classes_to_bground=True,
                 permute_instance_order=False, set_extras_to_void=False,
                 return_semantic_instance_tuple=False, resized_sz=None):
        """
        n_max_per_class: number of instances per non-background class
        class_subet: if None, use all classes.  Else, reduce the classes to this list set.
        map_other_classes_to_bground: if False, will error if classes in the training set are outside semantic_subset.
        permute_instance_order: randomly chooses the ordering of the instances (from 0 through
        n_max_per_class - 1) --> Does this every time the image is loaded.
        return_semantic_instance_tuple : Generally only for debugging; instead of returning an
        instance index as the target values, it'll return two targets: the semantic target and
        the instance number: [0, n_max_per_class)
        """
        self.id_is_in_eval = [not x for x in labels_table_cityscapes.ignore_in_eval]
        self.id_has_instances = labels_table_cityscapes.has_instances
        self.id_is_void = labels_table_cityscapes.is_void
        self.label_colors = labels_table_cityscapes.colors

        self.permute_instance_order = permute_instance_order
        self.map_other_classes_to_bground = map_other_classes_to_bground
        self.root = root
        self.split = split
        self._transform = transform
        self.n_max_per_class = n_max_per_class
        self.semantic_idxs_into_all_cityscapes = [0] + self.valid_instance_classes
        self.semantic_subset = semantic_subset or self.semantic_idxs_into_all_cityscapes
        self.n_semantic_classes = len(self.class_names)
        self._instance_to_semantic_mapping_matrix = None
        self.get_instance_to_semantic_mapping()
        self.n_classes = self._instance_to_semantic_mapping_matrix.size(0)
        self.set_extras_to_void = set_extras_to_void
        self.return_semantic_instance_tuple = return_semantic_instance_tuple
        self.resized_sz = resized_sz

        self.files = self.get_files()
        assert len(self) > 0, 'files[self.split={}] came up empty'.format(self.split)

    def update_n_max_per_class(self, n_max_per_class):
        self.n_max_per_class = n_max_per_class
        self.n_semantic_classes = len(self.class_names)
        self._instance_to_semantic_mapping_matrix = None
        self.get_instance_to_semantic_mapping(recompute=True)
        self.n_classes = self._instance_to_semantic_mapping_matrix.size(0)

    def get_files(self):
        files = collections.defaultdict(list)
        for split in ['train', 'val'] + ([] if self.split in ['train', 'val'] else [self.split]):

            images_base = os.path.join(self.root, 'leftImg8bit', split)
            annotations_base = os.path.join(self.root, 'gtFine_trainvaltest', 'gtFine', split)
            images = recursive_glob(rootdir=images_base, suffix='.png')
            for index, img_file in enumerate(images):
                img_file = img_file.rstrip()
                sem_lbl_file = os.path.join(annotations_base,
                                            img_file.split(os.sep)[-2],
                                            os.path.basename(img_file)[:-15] +
                                            'gtFine_labelIds.png')
                inst_lbl_file = os.path.join(annotations_base,
                                             img_file.split(os.sep)[-2],
                                             os.path.basename(img_file)[:-15] +
                                             'gtFine_instanceIds.png')
                assert osp.isfile(img_file), '{} does not exist'.format(img_file)
                assert osp.isfile(sem_lbl_file), '{} does not exist'.format(sem_lbl_file)
                assert osp.isfile(inst_lbl_file), '{} does not exist'.format(inst_lbl_file)
                # TODO(allie) -- allow functionality for permuting instance labels

                files[split].append({
                    'img': img_file,
                    'sem_lbl': sem_lbl_file,
                    'inst_lbl': inst_lbl_file,
                })
            assert len(files[split]) > 0, "No images found in directory {}".format(images_base)
        return files

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        data_file = self.files[self.split][index]
        img, lbl = self.load_and_process_cityscapes_files(img_file=data_file['img'],
                                                          sem_lbl_file=data_file['sem_lbl'],
                                                          inst_lbl_file=data_file['inst_lbl'],
                                                          return_semantic_instance_tuple=
                                                          self.return_semantic_instance_tuple)
        return img, lbl

    @staticmethod
    def get_semantic_names_and_idxs(semantic_subset):
        return dataset_utils.get_semantic_names_and_idxs(semantic_subset=semantic_subset,
                                                         full_set=ALL_CITYSCAPES_CLASS_NAMES)

    def remap_to_reduced_semantic_classes(self, lbl):
        return dataset_utils.remap_to_reduced_semantic_classes(
            lbl, reduced_class_idxs=self.semantic_idxs_into_all_cityscapes,
            map_other_classes_to_bground=self.map_other_classes_to_bground)

    def transform_img(self, img):
        return dataset_utils.transform_img(img, mean_bgr=self.mean_bgr, resized_sz=self.resized_sz)

    def transform_lbl(self, lbl, is_semantic):
        if DEBUG_ASSERT and self.resized_sz is not None:
            old_unique_classes = np.unique(lbl)
            logger.debug(
                'old_unique_classes ({}): {}'.format('semantic' if is_semantic else 'instance',
                                                     old_unique_classes))
            class_counts = [(lbl == c).sum() for c in old_unique_classes]
        else:
            old_unique_classes, class_counts = None, None
        lbl = dataset_utils.transform_lbl(lbl, resized_sz=self.resized_sz)
        if old_unique_classes is not None:
            new_unique_classes = np.unique(lbl.numpy())
            if not all([c in new_unique_classes for c in old_unique_classes]):
                classes_missing = [c for c in old_unique_classes if c not in new_unique_classes]
                class_indices_missing = [ci for ci, c in enumerate(old_unique_classes) if c in
                                         classes_missing]
                counts_missing = [class_counts[ci] for ci in class_indices_missing]
                # TODO(allie): set a better condition and raise to Info instead of Debug
                logger.debug(Warning(
                    'Resizing labels yielded fewer classes.  Missing classes {}, '
                    'totaling {} pixels'.format(classes_missing, counts_missing)))

        if DEBUG_ASSERT and is_semantic:
            classes = np.unique(lbl)
            lbl_np = lbl.numpy() if torch.is_tensor else lbl
            if not np.all(np.unique(lbl_np[lbl_np != 255]) < self.n_classes):
                print('after det', classes, np.unique(lbl))
                import ipdb;
                ipdb.set_trace()
                raise ValueError("Segmentation map contained invalid class values")
        return lbl

    def transform(self, img, lbl, is_semantic):
        img = self.transform_img(img)
        lbl = self.transform_lbl(lbl, is_semantic)
        return img, lbl

    def untransform(self, img, lbl):
        img = self.untransform_img(img)
        lbl = self.untransform_lbl(lbl)
        return img, lbl

    def untransform_lbl(self, lbl):
        lbl = dataset_utils.untransform_lbl(lbl)
        return lbl

    def untransform_img(self, img):
        img = dataset_utils.untransform_img(img, self.mean_bgr, original_size=None)
        return img

    def get_instance_semantic_labels(self):
        instance_semantic_labels = []
        for semantic_cls_idx, cls_name in enumerate(self.class_names):
            if semantic_cls_idx == 0:  # only one background instance
                instance_semantic_labels += [semantic_cls_idx]
            else:
                instance_semantic_labels += [semantic_cls_idx in range(self.n_max_per_class)]
        return instance_semantic_labels

    def get_instance_to_semantic_mapping(self, recompute=False):
        """ returns a binary matrix, where semantic_instance_mapping is N x S """
        if recompute or self._instance_to_semantic_mapping_matrix is None:
            self._instance_to_semantic_mapping_matrix = \
                dataset_utils.get_instance_to_semantic_mapping(self.n_max_per_class,
                                                               self.n_semantic_classes)
        return self._instance_to_semantic_mapping_matrix

    def combine_semantic_and_instance_labels(self, sem_lbl, inst_lbl):
        return dataset_utils.combine_semantic_and_instance_labels(
            sem_lbl, inst_lbl, n_max_per_class=self.n_max_per_class,
            set_extras_to_void=self.set_extras_to_void)

    def load_and_process_cityscapes_files(self, img_file, sem_lbl_file, inst_lbl_file,
                                          return_semantic_instance_tuple=False):
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load semantic label
        sem_lbl = PIL.Image.open(sem_lbl_file)
        # map to reduced class set
        sem_lbl = encode_segmap(np.array(sem_lbl, dtype=np.int32))  # used to be uint8; not sure why
        sem_lbl = sem_lbl.astype(np.int32)
        if self._transform:
            img = self.transform_img(img)
            sem_lbl[sem_lbl == -1] = 255
            num_neg_ones = np.sum(sem_lbl == -1)
            if num_neg_ones > 0:
                logger.debug('Found {} pixels = -1'.format(num_neg_ones))
            sem_lbl = self.transform_lbl(sem_lbl, is_semantic=True)
            sem_lbl[sem_lbl == 255] = -1
        # Handle instances
        if self.n_max_per_class == 1:
            lbl = sem_lbl
        else:
            inst_lbl = PIL.Image.open(inst_lbl_file)
            inst_lbl = np.array(inst_lbl, dtype=np.int32)
            if self._transform:
                inst_lbl[inst_lbl == -1] = 255
                inst_lbl = self.transform_lbl(inst_lbl, is_semantic=False)
                inst_lbl[inst_lbl == 255] = -1
            if self.permute_instance_order:
                inst_lbl = dataset_utils.permute_instance_order(inst_lbl, self.n_max_per_class)
            if return_semantic_instance_tuple:
                lbl = [sem_lbl, inst_lbl]
            else:
                try:
                    inst_lbl[sem_lbl == 0] = -1
                    sem_lbl[inst_lbl < 1000] = -1  # Need to map to an 'extras' class
                    inst_lbl[inst_lbl < 1000] = 0  # Not individual instances
                    inst_lbl[inst_lbl > 1000] -= (inst_lbl[inst_lbl > 1000] / 1000) * 1000
                    lbl = self.combine_semantic_and_instance_labels(sem_lbl, inst_lbl)
                except:
                    import ipdb
                    ipdb.set_trace()
                    raise

        return img, lbl

    def modify_length(self, modified_length):
        self.files[self.split] = self.files[self.split][:modified_length]

    def copy(self, modified_length=10):
        my_copy = CityscapesClassSegBase(root=self.root)
        for attr, val in self.__dict__.items():
            setattr(my_copy, attr, val)
        assert modified_length <= len(my_copy), "Can\'t create a copy with more examples than " \
                                                "the initial dataset"
        self.modify_length(modified_length)
        return my_copy

    def encode_segmap(self, mask):
        ignore_index = -1
        void_classes = self.void_classes,
        valid_instance_classes = self.valid_instance_classes
        semantic_only_classes = self.semantic_only_classes
        # Put all void classes to zero
        logger.debug('unique classes before mapping: {}'.format(np.unique(mask)))
        for id in range(len(CITYSCAPES_LABELS_TABLE)):
            mask[mask == id] = id_to_trainId(id)
        logger.debug('unique classes after mapping: {}'.format(np.unique(mask)))
        for voidc in void_classes:
            if voidc == ignore_index:
                continue
            mask[mask == voidc] = ignore_index
        logger.debug('unique classes after void mapping: {}'.format(np.unique(mask)))
        for semanticc in semantic_only_classes:
            mask[mask == semanticc] = 0
        logger.debug('unique classes after semantic_only mapping: {}'.format(np.unique(mask)))
        for validc in valid_instance_classes:
            mask[mask == validc] = valid_instance_classes.index(validc) + 1
        logger.debug('unique classes after valid mapping: {}'.format(np.unique(mask)))
        return mask

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]
