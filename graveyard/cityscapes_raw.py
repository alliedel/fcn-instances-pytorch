import collections
import os.path
from glob import glob

import PIL.Image
import numpy as np
from torch.utils import data

# TODO(allie): Allow some classes to get mapped onto background
# TODO(allie): Allow shuffling within the dataset here (instead of with train_loader)
# TODO(allie): Make CityscapesWithTransformations inherit from Raw


CITYSCAPES_MEAN_BGR = np.array([73.15835921, 82.90891754, 72.39239876])


class CityscapesRawBase(data.Dataset):

    def __init__(self, root, split='train'):
        self.root = root
        self.split = split
        self.files = self.get_files()

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        img, lbl = load_cityscapes_files(data_file['img'],
                                         data_file['sem_lbl'],
                                         data_file['inst_lbl'])
        return img, lbl

    def get_files(self):
        files = collections.defaultdict(list)
        split = self.split
        images_base = os.path.join(self.root, 'leftImg8bit', split)
        images = sorted(glob(os.path.join(images_base, '*', '*.png')))
        for index, img_file in enumerate(images):
            img_file = img_file.rstrip()
            sem_lbl_file = img_file.replace('leftImg8bit/',
                                            'gtFine_trainvaltest/gtFine/'
                                            ).replace('leftImg8bit.png', 'gtFine_labelIds.png')
            inst_lbl_file = sem_lbl_file.replace('labelIds', 'instanceIds')
            assert os.path.isfile(img_file), '{} does not exist'.format(img_file)
            assert os.path.isfile(sem_lbl_file), '{} does not exist'.format(sem_lbl_file)
            assert os.path.isfile(inst_lbl_file), '{} does not exist'.format(inst_lbl_file)

            files[split].append({'img': img_file, 'sem_lbl': sem_lbl_file, 'inst_lbl': inst_lbl_file})
        assert len(files[split]) > 0, "No images found in directory {}".format(images_base)
        return files


def load_cityscapes_files(img_file, sem_lbl_file, inst_lbl_file):
    img = PIL.Image.open(img_file)
    img = np.array(img, dtype=np.uint8)
    # load semantic label
    sem_lbl = np.array(PIL.Image.open(sem_lbl_file), dtype=np.int32)
    # load instance label
    inst_lbl = np.array(PIL.Image.open(inst_lbl_file), dtype=np.int32)
    return img, (sem_lbl, inst_lbl)
