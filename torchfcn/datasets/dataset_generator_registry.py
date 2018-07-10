from scripts.configurations import voc_cfg, cityscapes_cfg
from torchfcn.datasets import dataset_utils
from torchfcn.datasets import precomputed_file_transformations, runtime_transformations
from torchfcn.datasets import voc, synthetic, cityscapes
from torchfcn.utils.misc import pop_without_del
from torchfcn.utils.misc import value_as_string


def get_default_datasets_for_instance_counts(dataset_type):
    """
    Dataset before any of the precomputed_file_transformation, runtime_transformation runs on it
    """
    if dataset_type == 'voc':
        default_cfg = voc_cfg.get_default_config()
        precomputed_file_transformation, runtime_transformation = get_transformations(default_cfg)
        train_dataset, val_dataset = get_voc_datasets(
            dataset_path=default_cfg['dataset_path'], precomputed_file_transformation=precomputed_file_transformation,
            runtime_transformation=runtime_transformation)
    elif dataset_type == 'cityscapes':
        default_cfg = cityscapes_cfg.get_default_config()
        precomputed_file_transformation, runtime_transformation = get_transformations(default_cfg)
        train_dataset, val_dataset = get_cityscapes_datasets(default_cfg['dataset_path'],
                                                             precomputed_file_transformation=
                                                             precomputed_file_transformation,
                                                             runtime_transformation=runtime_transformation)
    elif dataset_type == 'synthetic':
        raise NotImplementedError('synthetic is different every time -- cannot save instance counts in between')
    else:
        raise ValueError
    transformer_tag = ''
    for tr in [precomputed_file_transformation, runtime_transformation]:
        attributes = tr.get_attribute_items() if tr is not None else {}.items()
        transformer_tag += '__'.join(['{}-{}'.format(k, value_as_string(v)) for k, v in attributes])
    return train_dataset, val_dataset, transformer_tag


def get_dataset(dataset_type, cfg, transform=True):
    if dataset_type == 'voc':
        precomputed_file_transformation, runtime_transformation = get_transformations(cfg)
        dataset_path = cfg['dataset_path']
        train_dataset, val_dataset = get_voc_datasets(
            dataset_path=dataset_path, precomputed_file_transformation=precomputed_file_transformation,
            runtime_transformation=runtime_transformation, transform=transform)
    elif dataset_type == 'cityscapes':
        dataset_path = cfg['dataset_path']
        precomputed_file_transformation, runtime_transformation = get_transformations(cfg)
        train_dataset, val_dataset = get_cityscapes_datasets(dataset_path, precomputed_file_transformation,
                                                             runtime_transformation, transform=transform)
    elif dataset_type == 'synthetic':
        precomputed_file_transformation, runtime_transformation = get_transformations(cfg)
        n_images_train = pop_without_del(cfg, 'n_images_train', None)
        n_images_val = pop_without_del(cfg, 'n_images_val', None)
        ordering = cfg['ordering']
        intermediate_write_path = cfg['dataset_path']
        train_dataset, val_dataset = get_synthetic_datasets(
            n_images_train=n_images_train, n_images_val=n_images_val, ordering=ordering,
            precomputed_file_transformation=precomputed_file_transformation,
            runtime_transformation=runtime_transformation,
            intermediate_write_path=intermediate_write_path, transform=transform)
    else:
        raise NotImplementedError('Generator for dataset type {} not implemented.')
    return train_dataset, val_dataset


def get_transformations(cfg):
    # Get transformation parameters
    semantic_subset = cfg['semantic_subset']
    if semantic_subset is not None:
        class_names, reduced_class_idxs = dataset_utils.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset, full_set=voc.ALL_VOC_CLASS_NAMES)
    else:
        reduced_class_idxs = None

    if cfg['dataset_instance_cap'] == 'match_model':
        n_inst_cap_per_class = cfg['n_instances_per_class']
    else:
        assert isinstance(cfg['n_instances_per_class'], int), 'n_instances_per_class was set to {} of type ' \
                                                              '{}'.format(cfg['n_instances_per_class'],
                                                                          type(cfg['n_instances_per_class']))
        n_inst_cap_per_class = cfg['dataset_instance_cap']

    precomputed_file_transformation = precomputed_file_transformations.precomputed_file_transformer_factory(
        ordering=cfg['ordering'])
    runtime_transformation = runtime_transformations.runtime_transformer_factory(
        resize=cfg['resize'], resize_size=cfg['resize_size'], mean_bgr=None, reduced_class_idxs=reduced_class_idxs,
        map_other_classes_to_bground=True, map_to_single_instance_problem=cfg['single_instance'],
        n_inst_cap_per_class=n_inst_cap_per_class)

    return precomputed_file_transformation, runtime_transformation


def get_voc_datasets(dataset_path, precomputed_file_transformation, runtime_transformation, transform=True):
    train_dataset = voc.TransformedVOC(root=dataset_path, split='train',
                                       precomputed_file_transformation=precomputed_file_transformation,
                                       runtime_transformation=runtime_transformation)
    val_dataset = voc.TransformedVOC(root=dataset_path, split='seg11valid',
                                     precomputed_file_transformation=precomputed_file_transformation,
                                     runtime_transformation=runtime_transformation)

    if not transform:
        for dataset in [train_dataset, val_dataset]:
            dataset.should_use_precompute_transform = False
            dataset.should_use_runtime_transform = False

    return train_dataset, val_dataset


def get_cityscapes_datasets(dataset_path, precomputed_file_transformation, runtime_transformation, transform=True):
    train_dataset = cityscapes.TransformedCityscapes(
        root=dataset_path, split='train',
        precomputed_file_transformation=precomputed_file_transformation,
        runtime_transformation=runtime_transformation)
    val_dataset = cityscapes.TransformedCityscapes(
        root=dataset_path, split='val',
        precomputed_file_transformation=precomputed_file_transformation,
        runtime_transformation=runtime_transformation)

    if not transform:
        for dataset in [train_dataset, val_dataset]:
            dataset.should_use_precompute_transform = False
            dataset.should_use_runtime_transform = False

    return train_dataset, val_dataset


def get_synthetic_datasets(n_images_train, n_images_val, ordering, precomputed_file_transformation,
                           runtime_transformation, intermediate_write_path='/tmp/', transform=True):
    if isinstance(precomputed_file_transformation,
                  precomputed_file_transformations.InstanceOrderingPrecomputedDatasetFileTransformation):
        precomputed_file_transformation = None  # Remove it, because we're going to order them when generating
        # the images instead.
    if precomputed_file_transformation is not None:
        raise ValueError('Cannot perform file transformations on the synthetic dataset.')

    train_dataset = synthetic.TransformedInstanceDataset(
        raw_dataset=synthetic.BlobExampleGenerator(n_images=n_images_train,
                                                   ordering=ordering,
                                                   intermediate_write_path=intermediate_write_path),
        raw_dataset_returns_images=True,
        runtime_transformation=runtime_transformation)
    val_dataset = synthetic.TransformedInstanceDataset(
        raw_dataset=synthetic.BlobExampleGenerator(n_images=n_images_val,
                                                   ordering=ordering,
                                                   intermediate_write_path=intermediate_write_path),
        raw_dataset_returns_images=True,
        runtime_transformation=runtime_transformation)

    if not transform:
        for dataset in [train_dataset, val_dataset]:
            dataset.should_use_precompute_transform = False
            dataset.should_use_runtime_transform = False

    return train_dataset, val_dataset
