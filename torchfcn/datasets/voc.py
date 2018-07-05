#!/usr/bin/env python

import os.path as osp

import numpy as np
from torch.utils import data

from torchfcn.datasets import dataset_utils, dataset_runtime_transformations

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


class RawVOCBase(data.Dataset):
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


class TransformedVOC(data.Dataset):
    """
    Has a raw dataset
    """

    def __init__(self, root, split, precomputed_file_transformation=None, runtime_transformation=None):
        self.raw_dataset = RawVOCBase(root, split=split)
        self.precomputed_file_transformation = precomputed_file_transformation
        self.runtime_transformation = runtime_transformation
        self.should_use_precompute_transform = True
        self.should_use_runtime_transform = True
        self.semantic_class_names = self.get_semantic_class_names()

    def get_semantic_class_names(self):
        """
        If we changed the semantic subset, we have to account for that change in the semantic class name list.
        """
        if self.should_use_runtime_transform and self.runtime_transformation is not None:
            transformation_list = self.runtime_transformation.transformer_sequence if isinstance(
                self.runtime_transformation, dataset_runtime_transformations.RuntimeDatasetTransformerSequence) else \
                [self.runtime_transformation]
            semantic_class_names = self.raw_dataset.semantic_class_names
            for transformer in transformation_list:
                if hasattr(transformer, 'transform_semantic_class_names'):
                    semantic_class_names = transformer.transform_semantic_class_names(
                        semantic_class_names)
            return semantic_class_names
        else:
            return self.raw_dataset.semantic_class_names

    def __getitem__(self, index):
        img, lbl = self.get_item(index,
                                 precomputed_file_transformation=self.precomputed_file_transformation,
                                 runtime_transformation=self.runtime_transformation)
        return img, lbl

    def __len__(self):  # explicit
        return len(self.raw_dataset)

    def get_item(self, index, precomputed_file_transformation=None, runtime_transformation=None):
        data_file = self.raw_dataset.files[index]  # files populated when RawVOCBase was instantiated
        img_file, sem_lbl_file, inst_lbl_file = data_file['img'], data_file['sem_lbl'], data_file['inst_lbl']

        # Get the right file
        if precomputed_file_transformation is not None:
            img_file, sem_lbl_file, inst_lbl_file = \
                precomputed_file_transformation.transform(img_file=img_file, sem_lbl_file=sem_lbl_file,
                                                          inst_lbl_file=inst_lbl_file)

        # Run data through transformation
        img, lbl = load_voc_files(img_file, sem_lbl_file, inst_lbl_file)
        if runtime_transformation is not None:
            img, lbl = runtime_transformation.transform(img, lbl)

        return img, lbl
