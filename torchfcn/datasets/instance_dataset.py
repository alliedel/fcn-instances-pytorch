import abc

from torch.utils import data
from .dataset_runtime_transformations import GenericSequenceRuntimeDatasetTransformer


class InstanceDatasetBase(data.Dataset):
    __metaclass__ = abc.ABC

    @property
    @abc.abstractmethod
    def semantic_class_names(self):
        pass

    # __getitem__(self, index) enforced by data.Dataset
    # __len__(self) enforced by data.Dataset


class TransformedInstanceDataset(InstanceDatasetBase):
    __metaclass__ = data.Dataset

    def __init__(self, raw_dataset, raw_type, precomputed_file_transformation=None,
                 runtime_transformation=None):
        assert raw_type in ['files', 'images'], ValueError
        if raw_type == 'images':
            assert precomputed_file_transformation is None, 'Cannot do precomputed file transformation on datasets ' \
                                                            'of type \'images\' (generated on the fly).'
        self.raw_type = raw_type
        self.raw_dataset = raw_dataset
        self.precomputed_file_transformation = precomputed_file_transformation
        self.runtime_transformation = runtime_transformation
        self.should_use_precompute_transform = True
        self.should_use_runtime_transform = True

    def __len__(self):  # explicit
        return len(self.raw_dataset)

    def __getitem__(self, index):
        img, lbl = self.get_item(index,
                                 precomputed_file_transformation=self.precomputed_file_transformation,
                                 runtime_transformation=self.runtime_transformation)
        return img, lbl

    @property
    def semantic_class_names(self):
        return self.get_semantic_class_names()

    def get_semantic_class_names(self):
        """
        If we changed the semantic subset, we have to account for that change in the semantic class name list.
        """
        if self.should_use_runtime_transform and self.runtime_transformation is not None:
            transformation_list = self.runtime_transformation.transformer_sequence if isinstance(
                self.runtime_transformation, GenericSequenceRuntimeDatasetTransformer) else \
                [self.runtime_transformation]
            semantic_class_names = self.raw_dataset.semantic_class_names
            for transformer in transformation_list:
                if hasattr(transformer, 'transform_semantic_class_names'):
                    semantic_class_names = transformer.transform_semantic_class_names(
                        semantic_class_names)
            return semantic_class_names
        else:
            return self.raw_dataset.semantic_class_names

    def load_files(self, img_file, sem_lbl_file, inst_lbl_file):
        # often self.raw_dataset.load_files(?)
        raise NotImplementedError

    def get_item_from_files(self, index, precomputed_file_transformation=None):
        data_file = self.raw_dataset.files[index]  # files populated when RawVOCBase was instantiated
        img_file, sem_lbl_file, inst_lbl_file = data_file['img'], data_file['sem_lbl'], data_file['inst_lbl']

        # Get the right file
        if precomputed_file_transformation is not None:
            img_file, sem_lbl_file, inst_lbl_file = \
                precomputed_file_transformation.transform(img_file=img_file, sem_lbl_file=sem_lbl_file,
                                                          inst_lbl_file=inst_lbl_file)

        # Run data through transformation
        img, lbl = self.load_files(img_file, sem_lbl_file, inst_lbl_file)
        return img, lbl

    def get_item(self, index, precomputed_file_transformation=None, runtime_transformation=None):
        if self.raw_type == 'files':
            img, lbl = self.get_item_from_files(index, precomputed_file_transformation)
        elif self.raw_type == 'images':
            img, lbl = self.raw_dataset.__getitem__(index)
            assert precomputed_file_transformation is None, 'Cannot do precomputed file transformation on datasets ' \
                                                            'of type \'images\' (generated on the fly).'
        else:
            raise NotImplementedError
        if runtime_transformation is not None:
            img, lbl = runtime_transformation.transform(img, lbl)

        return img, lbl
