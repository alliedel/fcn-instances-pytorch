from torchfcn.datasets import dataset_utils


class RuntimeDatasetTransformerBase(object):
    def transform(self, img, lbl):
        raise NotImplementedError

    def untransform(self, img, lbl):
        raise NotImplementedError


# noinspection PyTypeChecker
def runtime_transformer_factory(resize=None, resize_size=None, mean_bgr=None, reduced_class_idxs=None,
                                map_other_classes_to_bground=True, map_to_single_instance_problem=False,
                                n_inst_cap_per_class=None):
    # Basic transformation (numpy array to torch tensor; resizing and centering)
    transformer_sequence = [BasicRuntimeDatasetTransformer(resize=resize, resize_size=resize_size, mean_bgr=mean_bgr)]

    # Image transformations

    # Semantic label transformations
    if reduced_class_idxs is not None and len(reduced_class_idxs) > 0:
        transformer_sequence.append(SemanticSubsetRuntimeDatasetTransformer(
            reduced_class_idxs=reduced_class_idxs, map_other_classes_to_bground=map_other_classes_to_bground))

    # Instance label transformations
    transformer_sequence.append(SemanticAgreementForInstanceLabelsRuntimeDatasetTransformer())

    if n_inst_cap_per_class is not None:
        transformer_sequence.append(InstanceNumberCapRuntimeDatasetTransformer(n_inst_cap_per_class=n_inst_cap_per_class))
        
    if map_to_single_instance_problem:
        transformer_sequence.append(SingleInstanceMapperRuntimeDatasetTransformer())

    # Stitching them together in a sequence
    if len(transformer_sequence) == 0:
        return None
    elif len(transformer_sequence) == 1:
        return transformer_sequence[0]
    else:
        return GenericSequenceRuntimeDatasetTransformer(transformer_sequence=transformer_sequence)


class SemanticAgreementForInstanceLabelsRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    def transform(self, img, lbl):
        new_inst_lbl = self.impose_semantic_constraints_on_instance_label(lbl[0], lbl[1])
        return img, (lbl[0], new_inst_lbl)

    def untransform(self, img, lbl):
        print(Warning('It\'s not possible to recover the initial instance labels.  Returning the existing ones.'))
        return img, lbl

    @staticmethod
    def impose_semantic_constraints_on_instance_label(sem_lbl, inst_lbl):
        inst_lbl[inst_lbl == 0] = -1  # right now, we do this because we're not using instance id 0 for object classes.
        inst_lbl[sem_lbl == 0] = 0  # needed after we map other semantic classes to background.
        sem_lbl[inst_lbl == -1] = -1  # void semantic should always match void instance.
        return inst_lbl


class InstanceNumberCapRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    def __init__(self, n_inst_cap_per_class):
        self.n_inst_cap_per_class = n_inst_cap_per_class

    def transform(self, img, lbl):
        new_inst_lbl = lbl[1]
        new_inst_lbl[new_inst_lbl > self.n_inst_cap_per_class] = -1
        return img, (lbl[0], new_inst_lbl)

    def untransform(self, img, lbl):
        print(Warning('It\'s not possible to recover the initial instance labels (many-to-one mapping).  Returning the '
                      'existing ones.'))
        return img, lbl


class SingleInstanceMapperRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    def transform(self, img, lbl):
        new_inst_lbl = lbl[1]
        new_inst_lbl[new_inst_lbl != -1] = 1
        return img, (lbl[0], new_inst_lbl)

    def untransform(self, img, lbl):
        print(Warning('It\'s not possible to recover the initial instance labels (many-to-one mapping).  Returning the '
                      'existing ones.'))
        return img, lbl


class SemanticSubsetRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    def __init__(self, reduced_class_idxs, map_other_classes_to_bground=True):
        self.reduced_class_idxs = reduced_class_idxs
        self.map_other_classes_to_bground = map_other_classes_to_bground

    def transform(self, img, lbl):
        sem_fcn = lambda sem_lbl: dataset_utils.remap_to_reduced_semantic_classes(
            sem_lbl, reduced_class_idxs=self.reduced_class_idxs,
            map_other_classes_to_bground=self.map_other_classes_to_bground)
        lbl = (sem_fcn(lbl[0]), lbl[1])
        return img, lbl

    def untransform(self, img, lbl):
        raise NotImplementedError('Implement here if needed.')


class BasicRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    """
    Resizes, centers, and converts to torch tensor
    """

    def __init__(self, resize=True, resize_size=(512, 1024), mean_bgr=None):
        self.resize = resize
        self.resize_size = resize_size
        self.mean_bgr = mean_bgr

    def transform(self, img, lbl):
        return self.transform_img(img), self.transform_lbl(lbl)

    def untransform(self, img, lbl):
        return self.untransform_img(img), self.untransform_lbl(lbl)

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


class GenericSequenceRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    def __init__(self, transformer_sequence):
        """
        :param transformer_sequence:   list of functions of type transform(img, lbl)
                                                or RuntimeDatasetTransformerBase objects
        """
        self.transformer_sequence = transformer_sequence

    def transform(self, img, lbl):
        for transformer in self.transformer_sequence:
            if callable(transformer):
                img, lbl = transformer(img, lbl)
            elif isinstance(transformer, RuntimeDatasetTransformerBase):
                img, lbl = transformer.transform(img, lbl)
        return img, lbl

    def untransform(self, img, lbl):
        assert all([isinstance(transformer, RuntimeDatasetTransformerBase) for transformer in self.transformer_sequence]), \
            ValueError('If you want to call untransform, your transform functions must be placed in a '
                       'RuntimeDatasetTransformerBase class with an untransform function.')
        for transformer in self.transformer_sequence[::-1]:
            img, lbl = transformer.untransform(img, lbl)
        return img, lbl


def generate_transformer_from_functions(img_transform_function=None, sem_lbl_transform_function=None,
                                        inst_lbl_transform_function=None, img_untransform_function=None,
                                        sem_lbl_untransform_function=None, inst_lbl_untransform_function=None,
                                        lbl_transform_function=None, lbl_untransform_function=None):
    """
    Creates an image/label transformer object based on a list of functions. lbl_transform_function operates on both
    the semantic and instance labels.  If you need to do something to both the semantic and instance labels,
    you may want to create two separate transformers (I give this option so the instance lbl transformer can depend on
    the semantic lbl
    """

    if lbl_transform_function is not None:
        assert inst_lbl_transform_function is None and sem_lbl_transform_function is None

    if lbl_untransform_function is not None:
        assert inst_lbl_untransform_function is None and sem_lbl_untransform_function is None

    class RuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
        can_untransform = all([transform_fnc is None or untransform_fnc is not None
                               for transform_fnc, untransform_fnc
                               in zip([img_transform_function, lbl_transform_function,
                                       sem_lbl_transform_function, inst_lbl_transform_function],
                                      [img_untransform_function, lbl_untransform_function,
                                       sem_lbl_untransform_function, inst_lbl_untransform_function])])

        def transform(self, img, lbl):
            if img_transform_function is not None:
                img = img_transform_function(img)
            if lbl_transform_function is not None:
                lbl = lbl_transform_function(lbl)
            else:
                if sem_lbl_transform_function is not None:
                    lbl[0] = sem_lbl_transform_function(lbl[0])
                if inst_lbl_transform_function is not None:
                    lbl[1] = sem_lbl_transform_function(lbl[1])
            return img, lbl

        def untransform(self, img, lbl):
            if not self.can_untransform:
                raise ValueError('I don\'t have the untransform function available.')
            if img_untransform_function is not None:
                img = img_untransform_function(img)
            if lbl_transform_function is not None:
                lbl = lbl_transform_function(lbl)
            else:
                if sem_lbl_transform_function is not None:
                    lbl[0] = sem_lbl_transform_function(lbl[0])
                if inst_lbl_transform_function is not None:
                    lbl[1] = sem_lbl_transform_function(lbl[1])
            return img, lbl

    transformer = RuntimeDatasetTransformer()
    return transformer
