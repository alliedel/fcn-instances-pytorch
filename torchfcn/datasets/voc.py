#!/usr/bin/env python

import collections
import numpy as np
import os.path as osp
import PIL.Image
import shutil

import torch
from torch.utils import data
import tqdm

from . import dataset_utils
from torchfcn import instance_utils


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

    class_names = ALL_VOC_CLASS_NAMES
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False,
                 semantic_subset=None, map_other_classes_to_bground=True,
                 permute_instance_order=False, set_extras_to_void=False,
                 return_semantic_instance_tuple=None, semantic_only_labels=None,
                 n_instances_per_class=None, filter_images_by_semantic_subset=False,
                 file_index_subset=None, _im_a_copy=False, map_to_single_instance_problem=False,
                 collect_image_details=None, weight_by_instance=False, instance_counts_precomputed=None):
        """
        semantic_subset: if None, use all classes.  Else, reduce the classes to this list set.
        map_other_classes_to_bground: if False, will error if classes in the training set are outside semantic_subset.
        return_semantic_instance_tuple : Generally only for debugging; instead of returning an
        instance index as the target values, it'll return two targets: the semantic target and
        the instance number: [0, n_instances_per_class[sem_idx])
        """

        assert (file_index_subset is None) or (not filter_images_by_semantic_subset), \
            ValueError('Cannot specify modified indices and image filtering method')
        assert not (weight_by_instance and collect_image_details is False), ValueError
        self.collect_image_details = collect_image_details if collect_image_details is not None else weight_by_instance
        self.weight_by_instance = weight_by_instance
        self.map_to_single_instance_problem = map_to_single_instance_problem
        if return_semantic_instance_tuple is None:
            return_semantic_instance_tuple = True if not semantic_only_labels else False
        if semantic_only_labels is None:
            semantic_only_labels = False
        self.file_index_subset = file_index_subset  # None: all files.
        self.permute_instance_order = permute_instance_order
        if permute_instance_order:
            raise NotImplementedError
        self.map_other_classes_to_bground = map_other_classes_to_bground
        self.root = root
        self.split = split
        self._transform = transform
        self.semantic_subset = semantic_subset
        self.class_names, self.idxs_into_all_voc = dataset_utils.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset, full_set=ALL_VOC_CLASS_NAMES)
        self.n_semantic_classes = len(self.class_names)
        self._instance_to_semantic_mapping_matrix = None
        if n_instances_per_class is None:
            self.n_inst_per_class = [1 if cls_nm == 'background' else 1  # default to 1 per class
                                     for cls_idx, cls_nm in enumerate(self.class_names)]
        else:
            self.n_inst_per_class = n_instances_per_class
        assert xor(return_semantic_instance_tuple, semantic_only_labels)
        self.get_instance_to_semantic_mapping()
        self.n_classes = len(self.class_names)
        self.set_extras_to_void = set_extras_to_void
        self.return_semantic_instance_tuple = return_semantic_instance_tuple
        self.semantic_only_labels = semantic_only_labels
        self.filter_images_by_semantic_subset = filter_images_by_semantic_subset

        # VOC2011 and others are subset of VOC2012
        year = 2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC{}'.format(year))

        # Get files
        self.files = self.get_files(dataset_dir)

        # Get file subset
        if self.filter_images_by_semantic_subset and self.semantic_subset is not None:
            non_bground_idxs = [idx for idx, nm in zip(self.idxs_into_all_voc, self.class_names)
                                if nm is not 'background']
            self.modify_image_set(self.filter_by_semantic_subset(self.files[self.split], non_bground_idxs),
                                  index_from_originals=True)
        if self.collect_image_details:
            if instance_counts_precomputed is not None:
                if self.semantic_subset:
                    semantic_classes = range(self.n_semantic_classes)
                else:
                    semantic_classes = self.idxs_into_all_voc
                assert instance_counts_precomputed.shape == (len(self.files[split]), len(semantic_classes))
                self.instance_counts = instance_counts_precomputed
            else:
                self.instance_counts = self.collect_instance_counts(self.files[self.split], semantic_subset)
                np.save('/home/adelgior/data/datasets/VOC/instance_counts_semantic_subset-None.npy',
                        self.instance_counts)
        else:
            self.instance_counts = None
        if self.weight_by_instance:
            self.weights = self.instance_counts.sum(axis=1)
        else:
            self.weights = 1
        self.weighted_file_index_list = self.get_weighted_file_index_list()
        assert len(self) > 0, 'files[self.split={}] came up empty'.format(self.split)

    def get_file_index_list(self):
        return self.file_index_subset or range(len(self.files[self.split]))

    def get_weighted_file_index_list(self):
        file_index_list = self.get_file_index_list()
        if self.weights is None:
            raise Exception('Debug error: self.weights was not set')
        elif self.weights is 1:
            weighted_file_list = file_index_list
        else:
            weighted_file_list = []
            for file_index in file_index_list:
                weighted_file_list += [file_index for _ in range(int(self.weights[file_index]))]
        return weighted_file_list

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
                inst_absolute_lbl_file = osp.join(
                    dataset_dir, 'SegmentationObject/%s.png' % did)
                inst_lbl_file = osp.join(
                    dataset_dir, 'SegmentationObject/%s_per_sem_cls.png' % did)
                if not osp.isfile(inst_lbl_file):
                    if not osp.isfile(inst_absolute_lbl_file):
                        raise Exception('This image does not exist')
                    self.generate_per_sem_instance_file(inst_absolute_lbl_file, sem_lbl_file, inst_lbl_file)
                files[split].append({
                    'img': img_file,
                    'sem_lbl': sem_lbl_file,
                    'inst_absolute_lbl': inst_absolute_lbl_file,
                    'inst_lbl': inst_lbl_file,
                })

            assert len(files[split]) > 0, "No images found from list {}".format(imgsets_file)
        return files

    def __len__(self):
        return len(self.weighted_file_index_list)

    def __getitem__(self, index):
        file_indices = self.weighted_file_index_list
        data_file = self.files[self.split][file_indices[index]]
        img, lbl = self.load_and_process_voc_files(img_file=data_file['img'],
                                                   sem_lbl_file=data_file['sem_lbl'],
                                                   inst_lbl_file=data_file['inst_lbl'])
        return img, lbl

    def modify_image_set(self, index_list, index_from_originals=False):
        if index_from_originals and self.file_index_subset is not None:
            raise NotImplementedError
        if max(index_list) >= self.__len__():
            if self.file_index_subset is not None:
                self.file_index_subset = [self.file_index_subset[index] for index in index_list]
            else:
                self.file_index_subset = index_list
        else:
            raise ValueError('index list must be within 0 and {}, not {}'.format(self.__len__() - 1, max(index_list)))

    def generate_per_sem_instance_file(self, inst_absolute_lbl_file, sem_lbl_file, inst_lbl_file):
        print('Generating per-semantic instance file: {}'.format(inst_lbl_file))
        sem_lbl = self.load_img_as_dtype(sem_lbl_file, np.int32)
        unique_sem_lbls = np.unique(sem_lbl)
        if sum(unique_sem_lbls > 0) <= 1:  # only one semantic object type
            shutil.copyfile(inst_absolute_lbl_file, inst_lbl_file)
        else:
            inst_lbl = self.load_img_as_dtype(inst_absolute_lbl_file, np.int32)
            inst_lbl[inst_lbl == 255] = -1
            for sem_val in unique_sem_lbls[unique_sem_lbls > 0]:
                first_instance_idx = inst_lbl[sem_lbl == sem_val].min()
                inst_lbl[sem_lbl == sem_val] -= (first_instance_idx - 1)
            self.write_np_array_as_img(inst_lbl, inst_lbl_file)

    def write_np_array_as_img(self, arr, filename):
        im = PIL.Image.fromarray(arr.astype(np.uint8))
        im.save(filename)

    def load_img_as_dtype(self, img_file, dtype):
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=dtype)
        return img

    def transform_img(self, img):
        return dataset_utils.transform_img(img, self.mean_bgr, resized_sz=None)

    def transform_lbl(self, lbl):
        return dataset_utils.transform_lbl(lbl)

    def transform(self, img, lbl):
        img = self.transform_img(img)
        lbl = self.transform_lbl(lbl)
        return img, lbl

    def untransform(self, img, lbl):
        img = self.untransform_img(img)
        lbl = self.untransform_lbl(lbl)
        return img, lbl

    def untransform_img(self, img):
        return dataset_utils.untransform_img(img, self.mean_bgr, original_size=None)

    def untransform_lbl(self, lbl):
        return dataset_utils.untransform_lbl(lbl)

    def get_instance_to_semantic_mapping(self, recompute=False):
        """ returns a binary matrix, where semantic_instance_mapping is N x S """
        if recompute or self._instance_to_semantic_mapping_matrix is None:
            self._instance_to_semantic_mapping_matrix = \
                instance_utils.get_instance_to_semantic_mapping(self.n_inst_per_class,
                                                                self.n_semantic_classes)
        return self._instance_to_semantic_mapping_matrix

    def combine_semantic_and_instance_labels(self, sem_lbl, inst_lbl):
        raise NotImplementedError('we need to pass or create the instance config class to make this work properly')
        # return instance_utils.combine_semantic_and_instance_labels(
        #     sem_lbl, inst_lbl, semantic_instance_class_list=self.n_inst_per_class,
        #     instance_count_id_list=instance_utils.get_instance_count_id_list(self.semantic ... )
        #     set_extras_to_void=self.set_extras_to_void)

    def load_and_process_voc_files(self, img_file, sem_lbl_file, inst_lbl_file):
        img = self.load_img_as_dtype(img_file, np.uint8)
        # load semantic label
        sem_lbl = self.load_img_as_dtype(sem_lbl_file, np.int32)
        sem_lbl[sem_lbl == 255] = -1
        if self._transform:
            img, sem_lbl = self.transform(img, sem_lbl)
        # map to reduced class set
        sem_lbl = self.remap_to_reduced_semantic_classes(sem_lbl)

        # Handle instances
        if self.semantic_only_labels:
            lbl = sem_lbl
        else:
            inst_lbl = self.load_img_as_dtype(inst_lbl_file, np.int32)
            inst_lbl[inst_lbl == 255] = -1
            if self.map_to_single_instance_problem:
                inst_lbl[inst_lbl != -1] = 1
            inst_lbl[sem_lbl == -1] = -1
            inst_lbl = self.transform_lbl(inst_lbl)
            if self.return_semantic_instance_tuple:
                lbl = [sem_lbl, inst_lbl]
            else:
                lbl = self.combine_semantic_and_instance_labels(sem_lbl, inst_lbl)
        return img, lbl

    def remap_to_reduced_semantic_classes(self, lbl):
        return dataset_utils.remap_to_reduced_semantic_classes(
            lbl, reduced_class_idxs=self.idxs_into_all_voc,
            map_other_classes_to_bground=self.map_other_classes_to_bground)

    def filter_by_semantic_subset(self, sem_lbl_files, semantic_subset_vals):
        valid_indices = []
        for index, file in enumerate(sem_lbl_files):
            sem_lbl = self.load_img_as_dtype(file, np.int32)
            sem_lbl[sem_lbl == 255] = -1
            is_valid = False
            for semantic_val in semantic_subset_vals:
                if np.any(sem_lbl == semantic_val):
                    is_valid = True
                    break
            if is_valid:
                valid_indices.append(index)
        return valid_indices

    def copy(self, modified_length=10):
        my_copy = self.__class__(root=self.root, _im_a_copy=True)
        for attr, val in self.__dict__.items():
            setattr(my_copy, attr, val)
        assert modified_length <= len(my_copy), "Can\'t create a copy with more examples than " \
                                                "the initial dataset"
        my_copy.n_images = modified_length
        return my_copy

    def collect_instance_counts(self, files, semantic_classes=None):
        semantic_classes = semantic_classes or range(self.n_semantic_classes)
        instance_counts = np.ones((len(files), len(semantic_classes))) * np.nan
        file_index_list = self.get_file_index_list()
        for file_idx in tqdm.tqdm(file_index_list, total=len(file_index_list), desc='Analyzing VOC files', ncols=80,
                                  leave=False):
            for sem_idx, sem_val in enumerate(semantic_classes):
                data_file = self.files[self.split][file_idx]
                img, (sem_lbl, inst_lbl) = self.load_and_process_voc_files(img_file=data_file['img'],
                                                                           sem_lbl_file=data_file['sem_lbl'],
                                                                           inst_lbl_file=data_file['inst_lbl'])
                sem_locations_bool = sem_lbl == sem_val
                if torch.np.any(sem_locations_bool):
                    instance_counts[file_idx, sem_idx] = inst_lbl[sem_locations_bool].max()
                else:
                    instance_counts[file_idx, sem_idx] = 0
        return instance_counts


class VOC2011ClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', transform=False, **kwargs):
        super(VOC2011ClassSeg, self).__init__(
            root, split=split, transform=transform, **kwargs)


class VOC2012ClassSeg(VOCClassSegBase):
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False, **kwargs):
        super(VOC2012ClassSeg, self).__init__(
            root, split=split, transform=transform, **kwargs)


def xor(a, b):
    return (a and not b) or (not a and b)


    # class SBDClassSeg(VOCClassSegBase):
#
#     # XXX: It must be renamed to benchmark.tar to be extracted.
#     url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA
#
#     def __init__(self, root, split='train', transform=False, **kwargs):
#         super(SBDClassSeg, self).__init__(
#             root, split=split, transform=transform, **kwargs)
#         self.root = root
#         self.split = split
#         self._transform = transform
#
#         dataset_dir = osp.join(self.root, 'VOC/benchmark_RELEASE/dataset')
#         self.files = collections.defaultdict(list)
#         for split in ['train', 'val']:
#             imgsets_file = osp.join(dataset_dir, '%s.txt' % split)
#             for did in open(imgsets_file):
#                 did = did.strip()
#                 img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
#                 sem_lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
#                 self.files[split].append({
#                     'img': img_file,
#                     'sem_lbl': sem_lbl_file,
#                 })
#
#     def __getitem__(self, index):
#         data_file = self.files[self.split][index]
#         # load image
#         img_file = data_file['img']
#         img = PIL.Image.open(img_file)
#         img = np.array(img, dtype=np.uint8)
#         # load label
#         lbl_file = data_file['lbl']
#         mat = scipy.io.loadmat(lbl_file)
#         lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
#         lbl[lbl == 255] = -1
#         if self._transform:
#             return self.transform(img, lbl)
#         else:
#             return img, lbl


