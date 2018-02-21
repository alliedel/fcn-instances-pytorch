import collections
from glob import glob
import numpy as np
import os.path
import PIL.Image
import scipy.misc
from torch.utils import data

import dataset_utils
import labels_table_cityscapes


class CityscapesRawBase(data.Dataset):

    def __init__(self, root, split='train'):
        self.files = self.get_files()
        self.root = root
        self.split = split

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
        images = sorted(glob(os.path.join(images_base, '/*/*.png')))
        for index, img_file in enumerate(images):
            img_file = img_file.rstrip()
            sem_lbl_file = img_file.replace('leftImg8bit/',
                                            'gtFine_trainvaltest/gtFine/'
                                            ).replace('leftImg8bit', 'gtFine_labelIds.png')
            inst_lbl_file = sem_lbl_file.replace('labelIds', 'instanceIds')
            assert os.path.isfile(img_file), '{} does not exist'.format(img_file)
            assert os.path.isfile(sem_lbl_file), '{} does not exist'.format(sem_lbl_file)
            assert os.path.isfile(inst_lbl_file), '{} does not exist'.format(inst_lbl_file)

            files[split].append({'img': img_file, 'sem_lbl': sem_lbl_file, 'inst_lbl':
                                 inst_lbl_file})
        assert len(files[split]) > 0, "No images found in directory {}".format(images_base)
        return files


class DatasetTransformer(data.Dataset):
    mean_bgr = np.array([73.15835921, 82.90891754, 72.39239876])
    def __init__(self, resize=None):

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



def load_cityscapes_files(img_file, sem_lbl_file, inst_lbl_file):
    img = PIL.Image.open(img_file)
    img = np.array(img, dtype=np.uint8)
    # load semantic label
    sem_lbl = PIL.Image.open(sem_lbl_file).astype(np.int32)
    # load instance label
    inst_lbl = PIL.Image.open(inst_lbl_file).astype(np.int32)
    return img, (sem_lbl, inst_lbl)
