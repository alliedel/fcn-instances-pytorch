#!/usr/bin/env python

import collections
import os.path as osp
import PIL.Image

import numpy as np

import torch
from torch.utils import data

from . import dataset_utils
import scipy.misc as m
from torch.utils import data
import os
import torch
import numpy as np
import scipy.misc as m

DEBUG_ASSERT = True

"""CityscapesLoader

https://www.cityscapes-dataset.com

Data is derived from CityScapes, and can be downloaded from here:
https://www.cityscapes-dataset.com/downloads/

Many Thanks to @fvisin for the loader repo:
https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
"""

# TODO(allie): Allow for augmentations

# TODO(allie): Avoid so much code copying by making a semantic->instance wrapper class or base class

DEBUG_ASSERT = True

ALL_CITYSCAPES_CLASS_NAMES = np.array(
    ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
     'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
     'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
     'motorcycle', 'bicycle'])
CITYSCAPES_VOID_CLASSES = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
CITYSCAPES_VALID_CLASSES = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32,
                            33]
CITYSCAPES_IGNORE_INDEX = 250

CITYSCAPES_LABEL_COLORS = [  # [  0,   0,   0],
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [0, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32]]


class CityscapesClassSegBase(data.Dataset):
    mean_bgr = np.array([73.15835921, 82.90891754, 72.39239876])

    def __init__(self, root, split='train', transform=False, n_max_per_class=1,
                 semantic_subset=None, map_other_classes_to_bground=True,
                 permute_instance_order=True, set_extras_to_void=False,
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
        self.valid_classes = CITYSCAPES_VALID_CLASSES
        self.void_classes = CITYSCAPES_VOID_CLASSES
        self.label_colors = CITYSCAPES_LABEL_COLORS

        self.permute_instance_order = permute_instance_order
        self.map_other_classes_to_bground = map_other_classes_to_bground
        self.root = root
        self.split = split
        self._transform = transform
        self.n_max_per_class = n_max_per_class
        self.class_names, self.idxs_into_all_cityscapes = self.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset)
        self.n_semantic_classes = len(self.class_names)
        self._instance_to_semantic_mapping_matrix = None
        self.get_instance_to_semantic_mapping()
        self.n_classes = self._instance_to_semantic_mapping_matrix.size(0)
        self.set_extras_to_void = set_extras_to_void
        self.return_semantic_instance_tuple = return_semantic_instance_tuple
        self.resized_sz = resized_sz
        self.ignore_index = CITYSCAPES_IGNORE_INDEX

        dataset_dir = self.root
        self.files = self.get_files(dataset_dir)
        assert len(self) > 0, 'files[self.split={}] came up empty'.format(self.split)

    def update_n_max_per_class(self, n_max_per_class):
        self.n_max_per_class = n_max_per_class
        self.n_semantic_classes = len(self.class_names)
        self._instance_to_semantic_mapping_matrix = None
        self.get_instance_to_semantic_mapping(recompute=True)
        self.n_classes = self._instance_to_semantic_mapping_matrix.size(0)

    def get_files(self, dataset_dir):

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
            lbl, reduced_class_idxs=self.idxs_into_all_cityscapes,
            map_other_classes_to_bground=self.map_other_classes_to_bground)

    def transform_img(self, img):
        return dataset_utils.transform_img(img, mean_bgr=self.mean_bgr, resized_sz=self.resized_sz)

    def transform_lbl(self, lbl, is_semantic):
        lbl = dataset_utils.transform_lbl(lbl, resized_sz=self.resized_sz)
        if DEBUG_ASSERT and self.resized_sz is not None:
            classes = np.unique(lbl)
            if not np.all(classes == np.unique(lbl)):
                print("WARN: resizing labels yielded fewer classes")

        if DEBUG_ASSERT and is_semantic:
            classes = np.unique(lbl)
            lbl_np = lbl.numpy() if torch.is_tensor else lbl
            if not np.all(np.unique(lbl_np[lbl_np != self.ignore_index]) < self.n_classes):
                print('after det', classes, np.unique(lbl))
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
        sem_lbl = encode_segmap(np.array(sem_lbl, dtype=np.uint8))

        sem_lbl = np.array(sem_lbl, dtype=np.int32)
        sem_lbl[sem_lbl == 255] = -1
        if self._transform:
            img = self.transform_img(img)
            sem_lbl = self.transform_lbl(sem_lbl, is_semantic=True)
        # map to reduced class set
        sem_lbl = self.remap_to_reduced_semantic_classes(sem_lbl)
        # Handle instances
        if self.n_max_per_class == 1:
            lbl = sem_lbl
        else:
            inst_lbl = PIL.Image.open(inst_lbl_file)
            inst_lbl = np.array(inst_lbl, dtype=np.int32)
            inst_lbl[inst_lbl == 255] = -1
            if self._transform:
                inst_lbl = self.transform_lbl(inst_lbl, is_semantic=False)
            if self.permute_instance_order:
                inst_lbl = dataset_utils.permute_instance_order(inst_lbl, self.n_max_per_class)
            if return_semantic_instance_tuple:
                lbl = [sem_lbl, inst_lbl]
            else:
                try:
                    lbl = self.combine_semantic_and_instance_labels(sem_lbl, inst_lbl)
                except:
                    import ipdb;
                    ipdb.set_trace()
                    raise

        return img, lbl

    def encode_segmap(self, mask):
        return encode_segmap(mask, ignore_index=self.ignore_index, void_classes=self.void_classes,
                             valid_classes=self.valid_classes)

    def decode_segmap(self, temp):
        return decode_segmap(temp, self.n_classes, self.label_colors)

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


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


def encode_segmap(mask, ignore_index=CITYSCAPES_IGNORE_INDEX, void_classes=CITYSCAPES_VOID_CLASSES,
                  valid_classes=CITYSCAPES_VALID_CLASSES):
    # Put all void classes to zero
    for voidc in void_classes:
        mask[mask == voidc] = ignore_index
    for validc in valid_classes:
        mask[mask == validc] = valid_classes.index(validc)
    return mask


def decode_segmap(temp, n_classes, label_colours):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, n_classes):
        r[temp == l] = label_colours[l][0]
        g[temp == l] = label_colours[l][1]
        b[temp == l] = label_colours[l][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    return rgb
