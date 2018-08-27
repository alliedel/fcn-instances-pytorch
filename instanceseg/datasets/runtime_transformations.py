from instanceseg.utils import datasets
import inspect


DEBUG_ASSERT = True


class RuntimeDatasetTransformerBase(object):
    # def __init__(self):
    #    self.original_semantic_class_names = None

    def transform(self, img, lbl):
        raise NotImplementedError

    def untransform(self, img, lbl):
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


# noinspection PyTypeChecker
def runtime_transformer_factory(resize=None, resize_size=None, mean_bgr=None, reduced_class_idxs=None,
                                map_other_classes_to_bground=True, map_to_single_instance_problem=False,
                                n_inst_cap_per_class=None):
    # Basic transformation (numpy array to torch tensor; resizing and centering)
    transformer_sequence = []

    if resize:
        transformer_sequence.append(ResizeRuntimeDatasetTransformer(resize_size=resize_size if resize else None))

    transformer_sequence.append(BasicRuntimeDatasetTransformer(mean_bgr=mean_bgr))

    # Image transformations

    # Semantic label transformations
    if reduced_class_idxs is not None and len(reduced_class_idxs) > 0:
        transformer_sequence.append(SemanticSubsetRuntimeDatasetTransformer(
            reduced_class_idxs=reduced_class_idxs, map_other_classes_to_bground=map_other_classes_to_bground))

    # Instance label transformations

    if n_inst_cap_per_class is not None:
        transformer_sequence.append(InstanceNumberCapRuntimeDatasetTransformer(
            n_inst_cap_per_class=n_inst_cap_per_class))
        
    if map_to_single_instance_problem:
        transformer_sequence.append(SingleInstanceMapperRuntimeDatasetTransformer())

    # Some post-processing (should maybe be later(?))
    transformer_sequence.append(SemanticAgreementForInstanceLabelsRuntimeDatasetTransformer())

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


class ResizeRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    def __init__(self, resize_size=None):
        self.resize_size = resize_size

    def transform(self, img, lbl):
        img = datasets.resize_img(img, self.resize_size)
        if isinstance(lbl, tuple):
            assert len(lbl) == 2, 'Should be semantic, instance label tuple'
            lbl = tuple(datasets.resize_lbl(l, self.resize_size) for l in lbl)
        else:
            lbl = datasets.resize_lbl(lbl, self.resize_size)

        return img, lbl

    def untransform(self, img, lbl):
        raise NotImplementedError('Possible to implement (assuming we store original sizes? Haven\'t yet.')


class InstanceNumberCapRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    def __init__(self, n_inst_cap_per_class: int, instance_id_for_excluded_instances=-1):
        """
        :param n_inst_cap_per_class:
        :param instance_id_for_excluded_instances: usually you either want it to be 0 or -1.
        -1 will ensure it's not counted toward the losses; 0 will put it into the 'extras' channel (or the 'semantic'
        channel if we've mapped everything to the semantic problem instead)
        """
        assert n_inst_cap_per_class >= 0
        self.n_inst_cap_per_class = n_inst_cap_per_class
        self.instance_id_for_excluded_instances = instance_id_for_excluded_instances

    def transform(self, img, lbl):
        new_inst_lbl = lbl[1]
        new_inst_lbl[new_inst_lbl > self.n_inst_cap_per_class] = self.instance_id_for_excluded_instances
        return img, (lbl[0], new_inst_lbl)

    def untransform(self, img, lbl):
        print(Warning('It\'s not possible to recover the initial instance labels (many-to-one mapping).  Returning the '
                      'existing ones.'))
        return img, lbl


class SingleInstanceMapperRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    def __init__(self, background_values=(0,)):
        self.background_values = background_values

    def transform(self, img, lbl):
        new_inst_lbl = lbl[1]
        sem_lbl = lbl[0]
        if DEBUG_ASSERT:
            for bv in self.background_values:
                # Make sure the initial background value instance labels were either 0 or -1
                assert ((new_inst_lbl != 0)[sem_lbl == bv]).sum() == ((new_inst_lbl == -1)[sem_lbl == bv]).sum()
        # Set objects to instance value 1
        new_inst_lbl[new_inst_lbl > 1] = 1
        return img, (lbl[0], new_inst_lbl)

    def untransform(self, img, lbl):
        print(Warning('It\'s not possible to recover the initial instance labels (many-to-one mapping).  Returning the '
                      'existing ones.'))
        return img, lbl


class SemanticSubsetRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    def __init__(self, reduced_class_idxs, map_other_classes_to_bground=True):
        self.reduced_class_idxs = reduced_class_idxs
        self.map_other_classes_to_bground = map_other_classes_to_bground
        self.original_semantic_class_names = None

    def transform(self, img, lbl):
        sem_fcn = lambda sem_lbl: datasets.remap_to_reduced_semantic_classes(
            sem_lbl, reduced_class_idxs=self.reduced_class_idxs,
            map_other_classes_to_bground=self.map_other_classes_to_bground)
        lbl = (sem_fcn(lbl[0]), lbl[1])
        return img, lbl

    def untransform(self, img, lbl):
        raise NotImplementedError('Implement here if needed.')

    def transform_semantic_class_names(self, original_semantic_class_names):
        self.original_semantic_class_names = original_semantic_class_names
        return [self.original_semantic_class_names[idx] for idx in self.reduced_class_idxs]

    def untransform_semantic_class_names(self):
        return self.original_semantic_class_names


class BasicRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
    """
    centers and converts to torch tensor
    """

    def __init__(self, mean_bgr=None):
        self.mean_bgr = mean_bgr

    def transform(self, img, lbl):
        return self.transform_img(img), self.transform_lbl(lbl)

    def untransform(self, img, lbl):
        return self.untransform_img(img), self.untransform_lbl(lbl)

    def transform_img(self, img):
        return datasets.convert_img_to_torch_tensor(img, mean_bgr=self.mean_bgr)

    def transform_lbl(self, lbl):
        if isinstance(lbl, tuple):
            assert len(lbl) == 2, 'Should be semantic, instance label tuple'
            lbl = tuple(datasets.convert_lbl_to_torch_tensor(l) for l in lbl)
        else:
            lbl = datasets.convert_lbl_to_torch_tensor(lbl)
        return lbl

    def untransform_lbl(self, lbl):
        if isinstance(lbl, tuple):
            assert len(lbl) == 2, 'Should be semantic, instance label tuple'
            lbl = tuple(datasets.convert_torch_lbl_to_numpy(l) for l in lbl)
        else:
            lbl = datasets.convert_torch_lbl_to_numpy(lbl)
        return lbl

    def untransform_img(self, img):
        img = datasets.convert_torch_img_to_numpy(img, self.mean_bgr)
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

    def get_attribute_items(self):
        attributes = []
        for transformer in self.transformer_sequence:
            a = transformer.get_attribute_items()
            attributes += a
        return attributes


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

    class CustomRuntimeDatasetTransformer(RuntimeDatasetTransformerBase):
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

    transformer_type = CustomRuntimeDatasetTransformer
    return transformer_type
