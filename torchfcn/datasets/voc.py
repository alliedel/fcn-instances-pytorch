#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data

DEBUG_ASSERT = True

ALL_VOC_CLASS_NAMES = np.array([
    'background',  # 0
    'aeroplane',   # 1
    'bicycle',     # 2
    'bird',        # 3
    'boat',        # 4
    'bottle',      # 5
    'bus',         # 6
    'car',         # 7
    'cat',         # 8
    'chair',       # 9
    'cow',         # 10
    'diningtable', # 11
    'dog',         # 12
    'horse',       # 13
    'motorbike',   # 14
    'person',      # 15
    'potted plant',# 16
    'sheep',       # 17
    'sofa',        # 18
    'train',       # 19
    'tv/monitor',  # 20
])


class VOCClassSegBase(data.Dataset):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False, n_max_per_class=1,
                 semantic_subset=None, map_other_classes_to_bground=True):
        """
        n_max_per_class: number of instances per non-background class
        class_subet: if None, use all classes.  Else, reduce the classes to this list set.
        map_other_classes_to_bground: if False, will error if classes in the training set are outside semantic_subset.
        """
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

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        self.files = self.get_files(dataset_dir)

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
                        # for VOC2007 (and I assume other versions of VOC), the image names are
                        # different.  So if I generated a split for VOC2007, this allows me to
                        # use it.  Note it should break in the first iteration (year=2007),
                        # but maybe the image comes from another year (unlikely).
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
        assert len(self) > 0, 'files[self.split={}] came up empty'.format(self.split)
        return files

    @staticmethod
    def get_semantic_names_and_idxs(semantic_subset):
        if semantic_subset is None:
            names = ALL_VOC_CLASS_NAMES
            idxs_into_all_voc = range(len(ALL_VOC_CLASS_NAMES))
        else:
            idx_name_tuples = [(idx, cls) for idx, cls in enumerate(ALL_VOC_CLASS_NAMES)
                               if cls in semantic_subset]
            idxs_into_all_voc = [tup[0] for tup in idx_name_tuples]
            names = [tup[1] for tup in idx_name_tuples]
            assert 'background' in names, ValueError('You must include background in the list of '
                                                     'class names.')
            if len(idxs_into_all_voc) != len(semantic_subset):
                unrecognized_class_names = [cls for cls in semantic_subset if cls not in names]
                raise Exception('unrecognized class name(s): {}'.format(unrecognized_class_names))
        return names, idxs_into_all_voc

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load semantic label
        sem_lbl_file = data_file['sem_lbl']
        sem_lbl = PIL.Image.open(sem_lbl_file)
        sem_lbl = np.array(sem_lbl, dtype=np.int32)
        sem_lbl[sem_lbl == 255] = -1
        if self._transform:
            img, sem_lbl = self.transform(img, sem_lbl)
        # map to reduced class set
        sem_lbl = self.remap_to_reduced_semantic_classes(sem_lbl)

        # Handle instances
        import ipdb; ipdb.set_trace()
        if self.n_max_per_class == 1:
            lbl = sem_lbl
        else:
            inst_lbl_file = data_file['inst_lbl']
            inst_lbl = PIL.Image.open(inst_lbl_file)
            inst_lbl = np.array(inst_lbl, dtype=np.int32)
            inst_lbl[inst_lbl == 255] = -1
            inst_lbl = self.transform_lbl(inst_lbl)
            lbl = self.combine_semantic_and_instance_labels(sem_lbl, inst_lbl)

        return img, lbl

    def remap_to_reduced_semantic_classes(self, lbl):
        reduced_class_idxs = self.idxs_into_all_voc
        # Make sure all lbl classes can be mapped appropriately.
        if not self.map_other_classes_to_bground:
            original_classes_in_this_img = [i for i in range(lbl.min(), lbl.max()+1)
                                            if torch.sum(lbl == i) > 0 ]
            bool_unique_class_in_reduced_classes = [lbl_cls in reduced_class_idxs
                                                    for lbl_cls in original_classes_in_this_img
                                                    if lbl_cls != -1]
            if not all(bool_unique_class_in_reduced_classes):
                print(bool_unique_class_in_reduced_classes)
                import ipdb; ipdb.set_trace()
                raise Exception('Image has class labels outside the subset.\n Subset: {}\n'
                                'Classes in the image:{}'.format(reduced_class_idxs, original_classes_in_this_img))
        
        old_lbl = lbl.clone()
        lbl[...] = 0
        lbl[old_lbl == -1] = -1
        for new_idx, old_class_idx in enumerate(reduced_class_idxs):
            lbl[old_lbl == old_class_idx] = new_idx
        return lbl

    def transform_lbl(self, lbl):
        lbl = torch.from_numpy(lbl).long()
        return lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = self.transform_lbl(lbl)
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl

    def get_instance_semantic_labels(self):
        instance_semantic_labels = []
        for semantic_cls_idx, cls_name in enumerate(self.class_names):
            if semantic_cls_idx == 0:  # only one background instance
                instance_semantic_labels += [semantic_cls_idx]
            else:
                instance_semantic_labels += [semantic_cls_idx in range(self.n_max_per_class)]
        return instance_semantic_labels

    def get_instance_to_semantic_mapping(self, recompute=False):
        """
        returns a binary matrix, where semantic_instance_mapping is N x S
        (N = # instances, S = # semantic classes)
        semantic_instance_mapping[inst_idx, :] is a one-hot vector,
        and semantic_instance_mapping[inst_idx, sem_idx] = 1 iff that instance idx is an instance
        of that semantic class.
        """
        if recompute or self._instance_to_semantic_mapping_matrix is None:
            if self.n_max_per_class == 1:
                instance_to_semantic_mapping_matrix = torch.eye(self.n_semantic_classes,
                                                                self.n_semantic_classes)
            else:
                n_semantic_classes_with_background = len(self.class_names)
                n_instance_classes = \
                    1 + (n_semantic_classes_with_background - 1) * self.n_max_per_class
                instance_to_semantic_mapping_matrix = torch.zeros(
                    (n_instance_classes, n_semantic_classes_with_background)).float()

                semantic_instance_class_list = [0]
                for semantic_class in range(n_semantic_classes_with_background - 1):
                    semantic_instance_class_list += [semantic_class for _ in range(
                        self.n_max_per_class)]

                for instance_idx, semantic_idx in range(n_instance_classes):
                    instance_to_semantic_mapping_matrix[instance_idx,
                                                        semantic_idx] = 1
            self._instance_to_semantic_mapping_matrix = instance_to_semantic_mapping_matrix
        return self._instance_to_semantic_mapping_matrix

    def combine_semantic_and_instance_labels(self, sem_lbl, inst_lbl):
        """
        sem_lbl is size(img); inst_lbl is size(img).  inst_lbl is just the original instance
        image (inst_lbls at coordinates of person 0 are 0)
        """
        assert sem_lbl.shape == inst_lbl.shape
        if torch.np.any(inst_lbl >= self.n_max_per_class):
            raise Exception('There are more instances than the number you allocated for.')
            # if you don't want to raise an exception here, add a corresponding flag and use the
            # following line:
            # y = torch.min(inst_lbl, self.n_max_per_class)
        y = inst_lbl
        y += (sem_lbl - 1)*self.n_max_per_class
        y[y < 0] = 0  # background got 1+range(-self.n_max_per_class,0); all go to 0.

        instance_to_semantic_mapping = self.get_instance_to_semantic_mapping()
        if DEBUG_ASSERT:
            mapping_as_list_of_semantic_classes = torch.np.nonzero(
                torch.from_numpy(np.arange(self.n_semantic_classes)[
                                     instance_to_semantic_mapping[:,:] == 1
                                     ])).squeeze()
            assert mapping_as_list_of_semantic_classes.size() == (self.n_classes,)
            print('running assert statement now...')
            assert instance_to_semantic_mapping[y.ravel()] == sem_lbl.ravel()

        return y


class VOCClassSegBase2007(data.Dataset):
    class_names = ALL_VOC_CLASS_NAMES
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2007')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                sem_lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                inst_lbl_file = osp.join(
                    dataset_dir, 'SegmentationObject/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'sem_lbl': sem_lbl_file,
                    'inst_lbl': inst_lbl_file,
                })
                self.files[split].append({
                    'img': img_file,
                    'sem_lbl': sem_lbl_file,
                    'inst_lbl': inst_lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class VOC2011ClassSeg(VOCClassSegBase):
    def __init__(self, root, split='train', transform=False, n_max_per_class=1,
                 semantic_subset=None, **kwargs):
        super(VOC2011ClassSeg, self).__init__(
            root, split=split, transform=transform, n_max_per_class=n_max_per_class,
            semantic_subset=semantic_subset, **kwargs)
        pkg_root = osp.join(osp.dirname(osp.realpath(__file__)), '..')
        imgsets_file = osp.join(
            pkg_root, 'ext/fcn.berkeleyvision.org',
            'data/pascal/seg11valid.txt')
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})
            self.files['train'].append({'img': img_file, 'lbl': lbl_file})


class VOC2007ClassSegSingleImage(VOCClassSegBase2007):
    def __init__(self, root, split='train', transform=False):
        super(VOC2007ClassSegSingleImage, self).__init__(
            root, split=split, transform=transform)
        imgsets_file = osp.join(self.root, 'VOC/VOCdevkit/VOC2007/ImageSets/Segmentation/train.txt')
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2007')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})


class VOC2012ClassSeg(VOCClassSegBase):
    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False):
        super(VOC2012ClassSeg, self).__init__(
            root, split=split, transform=transform)


class SBDClassSeg(VOCClassSegBase):
    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = osp.join(self.root, 'VOC/benchmark_RELEASE/dataset')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl




