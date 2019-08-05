import collections
import os.path as osp

import PIL.Image
import numpy as np
import scipy.io

from instanceseg.utils import datasets
from instanceseg.utils.datasets import load_img_as_dtype
from instanceseg.datasets.voc import ALL_VOC_CLASS_NAMES

from . import voc


class SBDClassSeg(voc.VOCClassSegBase):
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

    def set_instance_cap(self, n_inst_cap_per_class=None):
        if not isinstance(n_inst_cap_per_class, int):
            raise NotImplementedError('Haven\'t implemented dif cap per semantic class. Please use an int.')
        self.n_inst_cap_per_class = n_inst_cap_per_class

    def reset_instance_cap(self):
        self.n_inst_cap_per_class = None

    def reduce_to_semantic_subset(self, semantic_subset):
        self.class_names, self.idxs_into_all_voc = datasets.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset, full_set=ALL_VOC_CLASS_NAMES)

    def clear_semantic_subset(self):
        self.class_names, self.idxs_into_all_voc = datasets.get_semantic_names_and_idxs(
            semantic_subset=None, full_set=ALL_VOC_CLASS_NAMES)

    def transform_img(self, img):
        return datasets.transform_img(img, self.mean_bgr, resized_sz=None)

    @staticmethod
    def transform_lbl(lbl):
        return datasets.transform_lbl(lbl)

    def transform(self, img, lbl):
        img = self.transform_img(img)
        lbl = self.transform_lbl(lbl)
        return img, lbl

    def untransform(self, img, lbl):
        img = self.untransform_img(img)
        lbl = self.untransform_lbl(lbl)
        return img, lbl

    def untransform_img(self, img):
        return datasets.untransform_img(img, self.mean_bgr, original_size=None)

    def untransform_lbl(self, lbl):
        return datasets.untransform_lbl(lbl)

    def combine_semantic_and_instance_labels(self, sem_lbl, inst_lbl):
        raise NotImplementedError('we need to pass or create the instance config class to make this work properly')

    def load_and_process_sem_lbl(self, sem_lbl_file):
        sem_lbl = load_img_as_dtype(sem_lbl_file, np.int32)
        sem_lbl[sem_lbl == 255] = -1
        if self._transform:
            sem_lbl = self.transform_lbl(sem_lbl)
        # map to reduced class set
        sem_lbl = self.remap_to_reduced_semantic_classes(sem_lbl)
        return sem_lbl

    def load_and_process_voc_files(self, img_file, sem_lbl_file, inst_lbl_file, gt_sem_inst_ordering_tuple_list=None):
        img = load_img_as_dtype(img_file, np.uint8)
        if self._transform:
            img = self.transform_img(img)

        # load semantic label
        sem_lbl = self.load_and_process_sem_lbl(sem_lbl_file)

        # load instance label
        if self.semantic_only_labels:
            lbl = sem_lbl
        else:
            inst_lbl = load_img_as_dtype(inst_lbl_file, np.int32)
            inst_lbl[inst_lbl == 255] = -1
            if self.map_to_single_instance_problem:
                inst_lbl[inst_lbl != -1] = 1
            if self._transform:
                inst_lbl = self.transform_lbl(inst_lbl)
            inst_lbl[sem_lbl == -1] = -1

            if self.n_inst_cap_per_class is not None:
                inst_lbl[inst_lbl > self.n_inst_cap_per_class] = -1

            inst_lbl[inst_lbl == 0] = -1  # sanity check
            inst_lbl[sem_lbl == 0] = 0  # needed for when we map other semantic classes to background.
            sem_lbl[inst_lbl == -1] = -1
            if self.return_semantic_instance_tuple:
                lbl = [sem_lbl, inst_lbl]
            else:
                lbl = self.combine_semantic_and_instance_labels(sem_lbl, inst_lbl)

        return img, lbl

    def remap_to_reduced_semantic_classes(self, sem_lbl):
        return datasets.remap_to_reduced_semantic_classes(sem_lbl, reduced_class_idxs=self.idxs_into_all_voc,
                                                          map_other_classes_to_bground=self.map_other_classes_to_bground)

