#!/usr/bin/env python

import collections
import os.path as osp
import PIL.Image

import numpy as np

import torch
from torch.utils import data

from . import dataset_utils


# TODO(allie): Allow for permuting the instance order at the beginning, and copying each filename
#  multiple times with the assigned permutation.  That way you can train in batches that have
# different permutations for the same image (may affect training if batched that way).
# You may also want to permute different semantic classes differently, though I'm pretty sure
# the network shouldn't be able to understand that's going on (semantic classes are handled
# separately)

DEBUG_ASSERT = True

ALL_VOC_CLASS_NAMES = np.array([
    'background',  # 0
    'aeroplane',  # 1
    'bicycle',  # 2
    'bird',  # 3
    'boat',  # 4
    'bottle',  # 5
    'bus',  # 6
    'car',  # 7
    'cat',  # 8
    'chair',  # 9
    'cow',  # 10
    'diningtable',  # 11
    'dog',  # 12
    'horse',  # 13
    'motorbike',  # 14
    'person',  # 15
    'potted plant',  # 16
    'sheep',  # 17
    'sofa',  # 18
    'train',  # 19
    'tv/monitor',  # 20
])


class VOCClassSegBase(data.Dataset):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False, n_max_per_class=1,
                 semantic_subset=None, map_other_classes_to_bground=True,
                 permute_instance_order=True, set_extras_to_void=False):
        """
        n_max_per_class: number of instances per non-background class
        class_subet: if None, use all classes.  Else, reduce the classes to this list set.
        map_other_classes_to_bground: if False, will error if classes in the training set are outside semantic_subset.
        permute_instance_order: randomly chooses the ordering of the instances (from 0 through
        n_max_per_class - 1) --> Does this every time the image is loaded.
        """

        self.permute_instance_order = permute_instance_order
        self.map_other_classes_to_bground = map_other_classes_to_bground
        self.root = root
        self.split = split
        self._transform = transform
        self.n_max_per_class = n_max_per_class
        self.class_names, self.idxs_into_all_voc = self.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset)
        self.n_semantic_classes = len(self.class_names)
        self._instance_to_semantic_mapping_matrix = None
        self.get_instance_to_semantic_mapping()
        self.n_classes = self._instance_to_semantic_mapping_matrix.size(0)
        self.set_extras_to_void = set_extras_to_void

        # VOC2011 and others are subset of VOC2012
        year = 2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC{}'.format(year))
        self.files = self.get_files(dataset_dir)
        assert len(self) > 0, 'files[self.split={}] came up empty'.format(self.split)

    def get_files(self, dataset_dir):
        files = collections.defaultdict(list)
        for split in ['train', 'val'] + ([] if self.split in ['train', 'val'] else [self.split]):
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                try:
                    img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                    assert osp.isfile(img_file)
                except AssertionError:
                    if not osp.isfile(img_file):
                        # VOC > 2007 has years in the name (VOC2007 doesn't).  Handling both.
                        for did_ext in ['{}_{}'.format(year, did) for year in range(2007, 2013)]:
                            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did_ext)
                            if osp.isfile(img_file):
                                did = did_ext
                                break
                        if not osp.isfile(img_file):
                            raise
                sem_lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                if not osp.isfile(sem_lbl_file):
                    raise Exception('This image does not exist')
                # TODO(allie) -- allow functionality for permuting instance labels
                inst_lbl_file = osp.join(
                    dataset_dir, 'SegmentationObject/%s.png' % did)
                files[split].append({
                    'img': img_file,
                    'sem_lbl': sem_lbl_file,
                    'inst_lbl': inst_lbl_file,
                })
            assert len(files[split]) > 0, "No images found from list {}".format(imgsets_file)
        return files

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        data_file = self.files[self.split][index]
        img, lbl = self.load_and_process_voc_files(img_file=data_file['img'],
                                                   sem_lbl_file=data_file['sem_lbl'],
                                                   inst_lbl_file=data_file['inst_lbl'])
        return img, lbl

    @staticmethod
    def get_semantic_names_and_idxs(semantic_subset):
        return dataset_utils.get_semantic_names_and_idxs(semantic_subset=semantic_subset,
                                                         full_set=ALL_VOC_CLASS_NAMES)

    def remap_to_reduced_semantic_classes(self, lbl):
        return dataset_utils.remap_to_reduced_semantic_classes(
            lbl, reduced_class_idxs=self.idxs_into_all_voc,
            map_other_classes_to_bground=self.map_other_classes_to_bground)

    @staticmethod
    def transform_lbl(lbl):
        lbl = torch.from_numpy(lbl).long()
        return lbl

    def transform_img(self, img):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img

    def transform(self, img, lbl):
        img = self.transform_img(img)
        lbl = self.transform_lbl(lbl)
        return img, lbl

    def untransform(self, img, lbl):
        img = self.untransform_img(img)
        lbl = lbl.numpy()
        return img, lbl

    def untransform_img(self, img):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
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

    def combine_semantic_and_instance_labels(self, sem_lbl, inst_lbl):
        return dataset_utils.combine_semantic_and_instance_labels(
            sem_lbl, inst_lbl, n_max_per_class=self.n_max_per_class,
            n_semantic_classes=self.n_semantic_classes, n_classes=self.n_classes,
            set_extras_to_void=self.set_extras_to_void)

    def load_and_process_voc_files(self, img_file, sem_lbl_file, inst_lbl_file):
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load semantic label
        sem_lbl = PIL.Image.open(sem_lbl_file)
        sem_lbl = np.array(sem_lbl, dtype=np.int32)
        sem_lbl[sem_lbl == 255] = -1
        if self._transform:
            img, sem_lbl = self.transform(img, sem_lbl)
        # map to reduced class set
        sem_lbl = self.remap_to_reduced_semantic_classes(sem_lbl)

        # Handle instances
        if self.n_max_per_class == 1:
            lbl = sem_lbl
        else:
            inst_lbl = PIL.Image.open(inst_lbl_file)
            inst_lbl = np.array(inst_lbl, dtype=np.int32)
            inst_lbl[inst_lbl == 255] = -1
            inst_lbl = self.transform_lbl(inst_lbl)
            if self.permute_instance_order:
                inst_lbl = dataset_utils.permute_instance_order(inst_lbl, self.n_max_per_class)

            lbl = self.combine_semantic_and_instance_labels(sem_lbl, inst_lbl)

        return img, lbl


class VOC2012ClassSeg(VOCClassSegBase):
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False, **kwargs):
        super(VOC2012ClassSeg, self).__init__(
            root, split=split, transform=transform, **kwargs)
