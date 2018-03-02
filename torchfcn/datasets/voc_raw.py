import collections
from glob import glob
import numpy as np
import os.path
import PIL.Image
from torch.utils import data

import dataset_utils

class_names = ['void'] + \
              ['background',
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
               ]
ids = [255] + list(range(len(class_names) - 1))
assert ids[class_names.index('background')] == 0
train_ids = [i if i != 255 else -1 for i in ids]
has_instances = [tid > 0 for tid in train_ids]
colors = [None for _ in class_names]
ignore_in_eval = [tid < 0 for tid in train_ids]
is_void = ignore_in_eval


class VOCRawBase(data.Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.files = self.get_files()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        img, lbl = load_voc_files(data_file['img'],
                                  data_file['sem_lbl'],
                                  data_file['inst_lbl'])
        return img, lbl

    def get_files(self):
        year = 2012
        dataset_dir = os.path.join(self.root, 'VOC/VOCdevkit/VOC{}'.format(year))
        files = collections.defaultdict(list)
        split = self.split
        imgsets_file = os.path.join(
            dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
        for did in open(imgsets_file):
            img_file = self.get_img_file(dataset_dir, did)
            img_file = img_file.rstrip()
            sem_lbl_file = img_file.replace('JPEGImages', 'SegmentationClass').replace(
                '.jpg', '.png')
            inst_lbl_file = sem_lbl_file.replace('SegmentationClass', 'SegmentationObject')
            assert os.path.isfile(img_file), '{} does not exist'.format(img_file)
            assert os.path.isfile(sem_lbl_file), '{} does not exist'.format(sem_lbl_file)
            assert os.path.isfile(inst_lbl_file), '{} does not exist'.format(inst_lbl_file)

            files[split].append({'img': img_file, 'sem_lbl': sem_lbl_file, 'inst_lbl':
                inst_lbl_file})
        assert len(files[split]) > 0, "No images found in directory {}".format(images_base)
        return files

    @staticmethod
    def get_img_file(dataset_dir, did):
        did = did.strip()
        img_file = os.path.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
        try:
            assert os.path.isfile(img_file)
        except AssertionError:
            if not os.path.isfile(img_file):
                # VOC > 2007 has years in the name (VOC2007 doesn't).  Handling both.
                for did_ext in ['{}_{}'.format(year, did) for year in range(2007, 2013)]:
                    img_file = os.path.join(dataset_dir, 'JPEGImages/%s.jpg' % did_ext)
                    if os.path.isfile(img_file):
                        did = did_ext
                        break
                if not os.path.isfile(img_file):
                    raise
        return img_file



class VOCWithTransformations(VOCRawBase):
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

    def __init__(self, root, split, resize=False, resize_size=None):
        super(VOCWithTransformations, self).__init__(root, split)
        self.resize = resize
        self.resize_size = resize_size

    def __getitem__(self, index):
        img, (sem_lbl, inst_lbl) = super(VOCWithTransformations, self).__getitem__(index)
        img = self.transform_img(img)
        lbl = (self.transform_lbl(sem_lbl), self.transform_lbl(inst_lbl))
        return img, lbl

    def transform_img(self, img):
        return dataset_utils.transform_img(img, mean_bgr=self.mean_bgr,
                                           resized_sz=self.resize_size)

    def transform_lbl(self, lbl):
        lbl = dataset_utils.transform_lbl(lbl, resized_sz=self.resize_size)
        return lbl

    def untransform_lbl(self, lbl):
        lbl = dataset_utils.untransform_lbl(lbl)
        return lbl

    def untransform_img(self, img):
        img = dataset_utils.untransform_img(img, self.mean_bgr, original_size=None)
        return img


def load_voc_files(img_file, sem_lbl_file, inst_lbl_file):
    img = PIL.Image.open(img_file)
    img = np.array(img, dtype=np.uint8)
    # load semantic label
    sem_lbl = np.array(PIL.Image.open(sem_lbl_file), dtype=np.int32)
    # load instance label
    inst_lbl = np.array(PIL.Image.open(inst_lbl_file), dtype=np.int32)
    return img, (sem_lbl, inst_lbl)
