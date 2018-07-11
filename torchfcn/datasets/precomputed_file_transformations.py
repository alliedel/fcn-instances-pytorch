from torchfcn.datasets import dataset_utils
import os.path as osp
import inspect


class PrecomputedDatasetFileTransformerBase(object):
    # def __init__(self):
    #    self.original_semantic_class_names = None

    def transform(self, img_file, sem_lbl_file, inst_lbl_file):
        ## Template:
        # new_sem_lbl_file = sem_lbl_file.replace(<ext>, <new_file_ext>)
        # if not osp.isfile(new_sem_lbl_file ):
        #   assert osp.isfile(sem_lbl_file)
        #   <create_new_sem_file()>

        # return img_file, new_sem_lbl_file, inst_lbl_file
        raise NotImplementedError

    def untransform(self, img_file, sem_lbl_file, inst_lbl_file):
        ## Template:
        # old_sem_lbl_file = sem_lbl_file.replace(<new_file_ext>, <ext>)
        # assert osp.isfile(old_sem_lbl_file)
        # return img_file, old_sem_lbl_file, inst_lbl_file
        raise NotImplementedError

    def get_attribute_items(self):
        attributes = inspect.getmembers(self, lambda a: not(inspect.isroutine(a)))
        attributes = [a for a in attributes if not(a[0].startswith('__') and a[0].endswith('__')) and not callable(a)]
        return attributes

    # def transform_semantic_class_names(self, original_semantic_class_names):
    # """ If exists, gets called whenever the dataset's semantic class names are queried. """
    #     self.original_semantic_class_names = original_semantic_class_names
    #     return fcn(self.original_semantic_class_names)
    #
    # def untransform_semantic_class_names(self):
    #     return self.original_semantic_class_names


def precomputed_file_transformer_factory(ordering=None):
    # Basic transformation (numpy array to torch tensor; resizing and centering)
    transformer_sequence = []

    # Image transformations

    # Semantic label transformations

    # Instance label transformations
    if ordering is not None:
        transformer_sequence.append(InstanceOrderingPrecomputedDatasetFileTransformation(ordering=ordering))

    # Stitching them together in a sequence
    if len(transformer_sequence) == 0:
        return None
    elif len(transformer_sequence) == 1:
        return transformer_sequence[0]
    else:
        return GenericSequencePrecomputedDatasetFileTransformer(transformer_sequence=transformer_sequence)


class InstanceOrderingPrecomputedDatasetFileTransformation(PrecomputedDatasetFileTransformerBase):

    def __init__(self, ordering=None):
        self.ordering = ordering  # 'lr', 'big_to_small'

    def transform(self, img_file, sem_lbl_file, inst_lbl_file):
        inst_lbl_file_unordered = inst_lbl_file
        if self.ordering is None:
            inst_lbl_file_ordered = inst_lbl_file_unordered
        elif self.ordering.lower() == 'lr':
            inst_lbl_file_ordered = inst_lbl_file_unordered.replace('.png', self.postfix + '.png')
            if not osp.isfile(inst_lbl_file_ordered):
                dataset_utils.generate_ordered_instance_file(inst_lbl_file_unordered,
                                                             sem_lbl_file, inst_lbl_file_ordered, ordering='lr',
                                                             increasing='True')
        elif self.ordering.lower() == 'big_to_small':
            inst_lbl_file_ordered = inst_lbl_file_unordered.replace('.png', self.postfix + '.png')
            if not osp.isfile(inst_lbl_file_ordered):
                dataset_utils.generate_ordered_instance_file(inst_lbl_file_unordered,
                                                             sem_lbl_file, inst_lbl_file_ordered,
                                                             ordering='size', increasing=False)
        else:
            raise ValueError('ordering={} not recognized'.format(self.ordering))
        return img_file, sem_lbl_file, inst_lbl_file_ordered

    def untransform(self, img_file, sem_lbl_file, inst_lbl_file):
        inst_lbl_file_ordered = inst_lbl_file
        if self.ordering is None:
            inst_lbl_file_unordered = inst_lbl_file_ordered
        elif self.ordering.lower() == 'lr':
            inst_lbl_file_unordered = inst_lbl_file_ordered.replace(self.postfix + '.png', '.png')
            if not osp.isfile(inst_lbl_file_ordered):
                raise FileNotFoundError('Cannot find the original file, {}'.format(inst_lbl_file_unordered))
        else:
            raise ValueError('ordering={} not recognized'.format(self.ordering))
        return img_file, sem_lbl_file, inst_lbl_file_unordered

    @property
    def postfix(self):
        return '_ordered_{}'.format(self.ordering)


class GenericSequencePrecomputedDatasetFileTransformer(PrecomputedDatasetFileTransformerBase):
    def __init__(self, transformer_sequence):
        """
        :param transformer_sequence:   list of functions of type transform(img, lbl)
                                                or RuntimeDatasetTransformerBase objects
        """
        self.transformer_sequence = transformer_sequence

    def transform(self, img_file, sem_lbl_file, inst_lbl_file):
        for transformer in self.transformer_sequence:
            if callable(transformer):
                img_file, sem_lbl_file, inst_lbl_file = transformer(img_file, sem_lbl_file, inst_lbl_file)
            elif isinstance(transformer, PrecomputedDatasetFileTransformerBase):
                img_file, sem_lbl_file, inst_lbl_file = transformer.transform(img_file, sem_lbl_file, inst_lbl_file)
        return img_file, sem_lbl_file, inst_lbl_file

    def untransform(self, img_file, sem_lbl_file, inst_lbl_file):
        assert all([isinstance(transformer, PrecomputedDatasetFileTransformerBase)
                    for transformer in self.transformer_sequence]), \
            ValueError('If you want to call untransform, your transform functions must be placed in a '
                       'PrecomputedDatasetFileTransformerBase class with an untransform function.')
        for transformer in self.transformer_sequence[::-1]:
            img_file, sem_lbl_file, inst_lbl_file = transformer.untransform(img_file, sem_lbl_file, inst_lbl_file)
        return img_file, sem_lbl_file, inst_lbl_file

    def get_attribute_items(self):
        attributes = []
        for transformer in self.transformer_sequence:
            a = transformer.get_attribute_items()
            attributes += a
        return attributes

