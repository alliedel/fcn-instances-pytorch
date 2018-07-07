#!/usr/bin/env python

import os.path as osp

import numpy as np
from torch.utils import data

from torchfcn.datasets import dataset_utils, dataset_runtime_transformations
from torchfcn.datasets.instance_dataset import InstanceDatasetBase, TransformedInstanceDataset

# TODO(allie): Allow for permuting the instance order at the beginning, and copying each filename
#  multiple times with the assigned permutation.  That way you can train in batches that have
# different permutations for the same image (may affect training if batched that way).
# You may also want to permute different semantic classes differently, though I'm pretty sure
# the network shouldn't be able to understand that's going on (semantic classes are handled
# separately)


MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])


def get_default_voc_root():
    other_options = [osp.abspath(osp.expanduser(p))
                     for p in ['~/afs_directories/kalman/data/datasets/VOC/VOCdevkit/VOC2012/']]
    VOC_ROOT = osp.abspath(osp.expanduser('~/data/datasets/VOC/VOCdevkit/VOC2012/'))
    if not osp.isdir(VOC_ROOT):
        for option in other_options:
            if osp.isdir(option):
                VOC_ROOT = option
                break
    return VOC_ROOT


VOC_ROOT = get_default_voc_root()

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


class RawVOCBase(InstanceDatasetBase):
    semantic_class_names = ALL_VOC_CLASS_NAMES

    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.files = self.get_files()
        self.semantic_class_names = ALL_VOC_CLASS_NAMES

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        img, lbl = load_voc_files(data_file['img'], data_file['sem_lbl'], data_file['inst_lbl'])
        return img, lbl

    def get_files(self):
        dataset_dir = self.root
        split = self.split
        file_list = get_raw_voc_files(dataset_dir, split)
        return file_list

    @property
    def n_semantic_classes(self):
        return len(self.semantic_class_names)


def get_raw_voc_files(dataset_dir, split):
    imgsets_file = osp.join(
        dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
    files = []
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
        inst_absolute_lbl_file = osp.join(
            dataset_dir, 'SegmentationObject/%s.png' % did)
        inst_lbl_file_unordered = osp.join(
            dataset_dir, 'SegmentationObject/%s_per_sem_cls.png' % did)
        if not osp.isfile(inst_lbl_file_unordered):
            if not osp.isfile(inst_absolute_lbl_file):
                raise Exception('This image does not exist')
            dataset_utils.generate_per_sem_instance_file(inst_absolute_lbl_file, sem_lbl_file, inst_lbl_file_unordered)

        files.append({
            'img': img_file,
            'sem_lbl': sem_lbl_file,
            'inst_lbl': inst_lbl_file_unordered,
            'inst_absolute_lbl': inst_absolute_lbl_file,
        })
    assert len(files) > 0, "No images found from list {}".format(imgsets_file)
    return files


def load_transformed_voc_files(self, img_file, sem_lbl_file, inst_lbl_file, transform=True):
    img, lbl = load_voc_files(img_file, sem_lbl_file, inst_lbl_file)
    if self.transformation is not None and transform:
        img, lbl = self.transformation.transform(img, lbl)
    return img, lbl


def load_voc_files(img_file, sem_lbl_file, inst_lbl_file=None, return_semantic_only=False):
    img = dataset_utils.load_img_as_dtype(img_file, np.uint8)
    sem_lbl = dataset_utils.load_img_as_dtype(sem_lbl_file, np.int32)
    sem_lbl[sem_lbl == 255] = -1

    # load instance label
    if return_semantic_only:
        assert inst_lbl_file is None, ValueError
        lbl = sem_lbl
    else:
        inst_lbl = dataset_utils.load_img_as_dtype(inst_lbl_file, np.int32)
        inst_lbl[inst_lbl == 255] = -1
        inst_lbl[sem_lbl == -1] = -1
        lbl = (sem_lbl, inst_lbl)

    return img, lbl


class TransformedVOC(TransformedInstanceDataset):
    """
    Has a raw dataset
    """

    def __init__(self, root, split, precomputed_file_transformation=None, runtime_transformation=None):
        raw_dataset = RawVOCBase(root, split=split)
        super(TransformedVOC, self).__init__(raw_dataset, precomputed_file_transformation, runtime_transformation)

    def load_files(self, img_file, sem_lbl_file, inst_lbl_file):
        return load_voc_files(img_file, sem_lbl_file, inst_lbl_file)
