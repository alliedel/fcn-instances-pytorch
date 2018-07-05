from . import dataset_runtime_transformations, dataset_precomputed_file_transformations
from . import dataset_utils, voc, synthetic


def get_dataset_with_transformations(dataset_type, split, root=None, transform=True, resize=None, resize_size=None,
                                     map_other_classes_to_bground=True, map_to_single_instance_problem=False,
                                     ordering=None, mean_bgr='default', semantic_subset=None,
                                     n_inst_cap_per_class=None, **kwargs):
    if kwargs:
        print('extra arguments while generating dataset: {}'.format(kwargs))
    if semantic_subset is not None:
        class_names, reduced_class_idxs = dataset_utils.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset, full_set=voc.ALL_VOC_CLASS_NAMES)
    else:
        reduced_class_idxs = None

    precomputed_file_transformation = dataset_precomputed_file_transformations.precomputed_file_transformer_factory(
        ordering=ordering)

    if mean_bgr == 'default':
        if dataset_type == 'voc':
            mean_bgr = None
        elif dataset_type == 'synthetic':
            mean_bgr = None
        else:
            print('Must set default mean_bgr for dataset {}'.format(dataset_type))

    runtime_transformation = dataset_runtime_transformations.runtime_transformer_factory(
        resize=resize, resize_size=resize_size, mean_bgr=mean_bgr, reduced_class_idxs=reduced_class_idxs,
        map_other_classes_to_bground=map_other_classes_to_bground,
        map_to_single_instance_problem=map_to_single_instance_problem, n_inst_cap_per_class=n_inst_cap_per_class)

    if dataset_type == 'voc':
        dataset = voc.TransformedVOC(root, split, precomputed_file_transformation=precomputed_file_transformation,
                                     runtime_transformation=runtime_transformation)
    else:
        raise NotImplementedError('I don\'t know dataset of type {}'.format(dataset_type))

    if not transform:
        dataset.should_use_precompute_transform = False
        dataset.should_use_runtime_transform = False

    return dataset
