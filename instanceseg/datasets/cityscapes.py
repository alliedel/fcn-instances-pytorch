import os.path as osp
from glob import glob

import PIL.Image
import numpy as np

from . import labels_table_cityscapes
from .cityscapes_transformations import CityscapesMapRawtoTrainIdPrecomputedFileDatasetTransformer, \
    ConvertLblstoPModePILImages
from instanceseg.datasets.precomputed_file_transformations import GenericSequencePrecomputedDatasetFileTransformer
from instanceseg.datasets.instance_dataset import InstanceDatasetBase, TransformedInstanceDataset


CITYSCAPES_MEAN_BGR = np.array([73.15835921, 82.90891754, 72.39239876])


def get_default_cityscapes_root():
    other_options = [osp.abspath(osp.expanduser(p))
                     for p in ['~/afs_directories/kalman/data/cityscapes/']]
    cityscapes_root = osp.abspath(osp.expanduser('data/cityscapes/'))
    if not osp.isdir(cityscapes_root):
        for option in other_options:
            if osp.isdir(option):
                cityscapes_root = option
                break
    return cityscapes_root


CITYSCAPES_ROOT = get_default_cityscapes_root()


class RawCityscapesBase(InstanceDatasetBase):

    original_semantic_class_names = labels_table_cityscapes.class_names  # by id (not trainId)
    precomputed_file_transformer = GenericSequencePrecomputedDatasetFileTransformer(
        [CityscapesMapRawtoTrainIdPrecomputedFileDatasetTransformer(),
         ConvertLblstoPModePILImages()])

    def __init__(self, root, split):
        self.root = root
        self.split = split
        self.files = self.get_files()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data_file = self.files[index]
        img, lbl = load_cityscapes_files(data_file['img'], data_file['sem_lbl'], data_file['inst_lbl'])
        return img, lbl

    def get_files(self):
        dataset_dir = self.root
        split = self.split
        orig_file_list = get_raw_cityscapes_files(dataset_dir, split)
        if self.precomputed_file_transformer is not None:
            file_list = []
            for i, data_files in enumerate(orig_file_list):
                img_file, sem_lbl_file, raw_inst_lbl_file = self.precomputed_file_transformer.transform(
                    img_file=data_files['img'],
                    sem_lbl_file=data_files['sem_lbl'],
                    inst_lbl_file=data_files['inst_lbl'])
                file_list.append({
                    'img': img_file,
                    'sem_lbl': sem_lbl_file,
                    'inst_lbl': raw_inst_lbl_file,
                })
        else:
            file_list = orig_file_list
        return file_list

    @property
    def semantic_class_names(self):
        return self.__class__.get_semantic_class_names()

    @classmethod
    def get_semantic_class_names(cls):
        """
        If we changed the semantic subset, we have to account for that change in the semantic class name list.
        """
        if cls.precomputed_file_transformer is not None:
            transformation_list = cls.precomputed_file_transformer.transformer_sequence if isinstance(
                cls.precomputed_file_transformer, GenericSequencePrecomputedDatasetFileTransformer) else \
                [cls.precomputed_file_transformer]
            semantic_class_names = cls.original_semantic_class_names
            for transformer in transformation_list:
                if hasattr(transformer, 'transform_semantic_class_names'):
                    semantic_class_names = transformer.transform_semantic_class_names(
                        semantic_class_names)
        else:
            semantic_class_names = cls.original_semantic_class_names
        assert AssertionError('There must be a bug somewhere.  The first semantic class name should always be '
                              'background.')
        return semantic_class_names

    @property
    def n_semantic_classes(self):
        return len(self.semantic_class_names)


def get_raw_cityscapes_files(dataset_dir, split):
    files = []
    images_base = osp.join(dataset_dir, 'leftImg8bit', split)
    images = sorted(glob(osp.join(images_base, '*', '*.png')))
    for index, img_file in enumerate(images):
        img_file = img_file.rstrip()
        sem_lbl_file = img_file.replace('leftImg8bit/', 'gtFine/').replace(
            'leftImg8bit.png', 'gtFine_labelIds.png')
        raw_inst_lbl_file = sem_lbl_file.replace('labelIds', 'instanceIds')
        assert osp.isfile(img_file), '{} does not exist'.format(img_file)
        assert osp.isfile(sem_lbl_file), '{} does not exist'.format(sem_lbl_file)
        assert osp.isfile(raw_inst_lbl_file), '{} does not exist'.format(raw_inst_lbl_file)

        files.append({
            'img': img_file,
            'sem_lbl': sem_lbl_file,
            'inst_lbl': raw_inst_lbl_file,
        })
    assert len(files) > 0, "No images found in directory {}".format(images_base)
    return files


def load_cityscapes_files(img_file, sem_lbl_file, inst_lbl_file):
    img = PIL.Image.open(img_file)
    img = np.array(img, dtype=np.uint8)
    # load semantic label
    sem_lbl = np.array(PIL.Image.open(sem_lbl_file), dtype=np.int32)
    # load instance label
    inst_lbl = np.array(PIL.Image.open(inst_lbl_file), dtype=np.int32)
    return img, (sem_lbl, inst_lbl)


class TransformedCityscapes(TransformedInstanceDataset):
    """
    Has a raw dataset
    """

    def __init__(self, root, split, precomputed_file_transformation=None, runtime_transformation=None):
        raw_dataset = RawCityscapesBase(root, split=split)
        super(TransformedCityscapes, self).__init__(raw_dataset=raw_dataset, raw_dataset_returns_images=False,
                                                    precomputed_file_transformation=precomputed_file_transformation,
                                                    runtime_transformation=runtime_transformation)

    def load_files(self, img_file, sem_lbl_file, inst_lbl_file):
        return load_cityscapes_files(img_file, sem_lbl_file, inst_lbl_file)

