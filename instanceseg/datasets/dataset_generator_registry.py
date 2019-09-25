from instanceseg.datasets import precomputed_file_transformations, runtime_transformations
from instanceseg.datasets import synthetic, cityscapes
from instanceseg.datasets.panoptic_dataset_base import get_transformer_identifier_tag
from instanceseg.utils import datasets
from instanceseg.utils.misc import pop_without_del, TermColors
from scripts.configurations import cityscapes_cfg
import numpy as np


def get_default_datasets_for_instance_counts(dataset_type, splits=('train', 'val')):
    """
    Dataset before any of the precomputed_file_transformation, runtime_transformation runs on it
    """
    if dataset_type == 'voc':
        raise NotImplementedError
    elif dataset_type == 'cityscapes':
        default_cfg = cityscapes_cfg.get_default_train_config()
        default_cfg[
            'n_instances_per_class'] = None  # don't want to cap instances when running stats
        precomputed_file_transformation, runtime_transformation = get_transformations(default_cfg)
        default_datasets = {
            split: get_cityscapes_dataset(
                default_cfg['dataset_path'],
                precomputed_file_transformation=precomputed_file_transformation,
                runtime_transformation=runtime_transformation, split=split) for split in splits
        }
    elif dataset_type == 'synthetic':
        # return None, None, None
        raise NotImplementedError('synthetic is different every time -- cannot save instance counts in between')
    else:
        raise ValueError
    transformer_tag = get_transformer_identifier_tag(precomputed_file_transformation,
                                                     runtime_transformation)
    return default_datasets, transformer_tag


def get_dataset(dataset_type, cfg, split, transform=True):
    if dataset_type == 'voc':
        raise NotImplementedError
    elif dataset_type == 'cityscapes':
        dataset_path = cfg['dataset_path']
        precomputed_file_transformation, runtime_transformation = get_transformations(
            cfg, cityscapes.CityscapesWithOurBasicTrainIds(dataset_path, split=split).semantic_class_names)
        dataset = get_cityscapes_dataset(dataset_path, precomputed_file_transformation, runtime_transformation,
                                         split=split, transform=transform)
    elif dataset_type == 'synthetic':
        semantic_subset = cfg['semantic_subset']
        precomputed_file_transformation, runtime_transformation = get_transformations(
            cfg, synthetic.ALL_BLOB_CLASS_NAMES)
        ordering = cfg['ordering']
        intermediate_write_path = cfg['dataset_path']
        n_instances_per_img = cfg['synthetic_generator_n_instances_per_semantic_id']
        img_size = cfg['resize_size']
        portrait = cfg['portrait']
        blob_size = cfg['blob_size']
        dataset = get_synthetic_dataset(
            split=split,
            n_images=pop_without_del(cfg, 'n_images_{}'.format(split), None),
            ordering=ordering,
            n_instances_per_img=n_instances_per_img,
            precomputed_file_transformation=precomputed_file_transformation,
            runtime_transformation=runtime_transformation,
            semantic_subset_to_generate=semantic_subset,
            img_size=img_size,
            portrait=portrait,
            intermediate_write_path=intermediate_write_path, transform=transform,
            blob_size=blob_size)
    else:
        raise NotImplementedError('Generator for dataset type {} not implemented.'.format(dataset_type))
    return dataset


def get_transformations(cfg, original_semantic_class_names=None):
    # Get transformation parameters
    semantic_subset = cfg['semantic_subset']
    if semantic_subset is not None:
        assert original_semantic_class_names is not None
        class_names, reduced_class_idxs = datasets.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset, full_set=original_semantic_class_names)
    else:
        reduced_class_idxs = None

    if cfg['dataset_instance_cap'] == 'match_model':
        n_inst_cap_per_class = cfg['n_instances_per_class']
    else:
        assert isinstance(cfg['n_instances_per_class'],
                          int), 'n_instances_per_class was set to {} of type ' \
                                '{}'.format(cfg['n_instances_per_class'],
                                            type(cfg['n_instances_per_class']))
        n_inst_cap_per_class = cfg['dataset_instance_cap']

    precomputed_file_transformation = \
        precomputed_file_transformations.precomputed_file_transformer_factory(
            ordering=cfg['ordering'])

    cfg_key = 'instance_id_for_excluded_instances'
    if cfg_key not in cfg:
        print(TermColors.WARNING + 'Setting value in cfg for what I believe is backwards compatibility: {}={}'.format(
            cfg_key, None) + TermColors.ENDC)
        cfg[cfg_key] = None

    runtime_transformation = runtime_transformations.runtime_transformer_factory(
        resize=cfg['resize'], resize_size=cfg['resize_size'], mean_bgr=None,
        reduced_class_idxs=reduced_class_idxs,
        map_other_classes_to_bground=True, map_to_single_instance_problem=
        cfg['single_instance'] or cfg['map_to_semantic'],
        n_inst_cap_per_class=n_inst_cap_per_class,
        instance_id_for_excluded_instances=cfg['instance_id_for_excluded_instances'])

    return precomputed_file_transformation, runtime_transformation


def get_cityscapes_dataset(dataset_path, precomputed_file_transformation, runtime_transformation, split,
                           transform=True):
    # assert split in ['train', 'val']
    dataset = cityscapes.TransformedCityscapes(
        root=dataset_path, split=split,
        precomputed_file_transformation=precomputed_file_transformation,
        runtime_transformation=runtime_transformation)
    if not transform:
        dataset.should_use_precompute_transform = False
        dataset.should_use_runtime_transform = False

    return dataset


def get_synthetic_dataset(n_images, ordering, n_instances_per_img,
                          precomputed_file_transformation, runtime_transformation, split,
                          intermediate_write_path='/tmp/',
                          semantic_subset_to_generate=None, portrait=None, img_size=None,
                          transform=True, random_seed=None, blob_size=None):
    assert split in ('train', 'val', 'test')
    if split == 'test' or split == 'val' and random_seed is None:
        random_seed = np.random.randint(100)
    if isinstance(precomputed_file_transformation,
                  precomputed_file_transformations.InstanceOrderingPrecomputedDatasetFileTransformation):
        assert precomputed_file_transformation.ordering == ordering, \
            AssertionError('We planned to synthetically generate ordering {}, but tried to replace '
                           'ordering {}'.format(ordering, precomputed_file_transformation.ordering))
        precomputed_file_transformation = None  # Remove it, because we're going to order them when generating
        # the images instead.
    if precomputed_file_transformation is not None:
        raise ValueError('Cannot perform file transformations on the synthetic dataset.')

    synthetic_kwargs = dict(ordering=ordering, intermediate_write_path=intermediate_write_path,
                            semantic_subset_to_generate=semantic_subset_to_generate,
                            n_instances_per_img=n_instances_per_img, img_size=img_size,
                            portrait=portrait, random_seed=random_seed, blob_size=blob_size)
    dataset = synthetic.TransformedPanopticDataset(
        raw_dataset=synthetic.BlobExampleGenerator(n_images=n_images, **synthetic_kwargs),
        raw_dataset_returns_images=True, runtime_transformation=runtime_transformation)

    if not transform:
        dataset.should_use_precompute_transform = False
        dataset.should_use_runtime_transform = False

    return dataset
