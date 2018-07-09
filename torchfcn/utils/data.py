import os

import numpy as np
import torch
import torch.utils.data

from torchfcn.datasets import samplers, dataset_utils, voc, cityscapes, synthetic, \
    dataset_precomputed_file_transformations, dataset_runtime_transformations
from torchfcn.utils.scripts import DEBUG_ASSERTS
from torchfcn.utils.samplers import get_configured_sampler


REGISTERED_DATASET_TYPES = ['cityscapes', 'voc', 'synthetic']


def get_datasets_with_transformations(dataset_type, cfg, transform=True):
    # Get transformation parameters
    semantic_subset = cfg['semantic_subset']
    if semantic_subset is not None:
        class_names, reduced_class_idxs = dataset_utils.get_semantic_names_and_idxs(
            semantic_subset=semantic_subset, full_set=voc.ALL_VOC_CLASS_NAMES)
    else:
        reduced_class_idxs = None
    try:
        mean_bgr = {'voc': None, 'cityscapes': None, 'synthetic': None}[dataset_type]
    except KeyError:
        raise Exception('Must set default mean_bgr for dataset {}'.format(dataset_type))

    if cfg['dataset_instance_cap'] == 'match_model':
        n_inst_cap_per_class = cfg['n_instances_per_class']
    else:
        assert isinstance(cfg['n_instances_per_class'], int), 'n_instances_per_class was set to {} of type ' \
                                                              '{}'.format(cfg['n_instances_per_class'],
                                                                          type(cfg['n_instances_per_class']))
        n_inst_cap_per_class = cfg['dataset_instance_cap']

    precomputed_file_transformation = dataset_precomputed_file_transformations.precomputed_file_transformer_factory(
        ordering=cfg['ordering'])
    runtime_transformation = dataset_runtime_transformations.runtime_transformer_factory(
        resize=cfg['resize'], resize_size=cfg['resize_size'], mean_bgr=mean_bgr, reduced_class_idxs=reduced_class_idxs,
        map_other_classes_to_bground=True, map_to_single_instance_problem=cfg['single_instance'],
        n_inst_cap_per_class=n_inst_cap_per_class)

    if dataset_type == 'voc':
        train_dataset = voc.TransformedVOC(root=cfg['dataset_path'], split='train',
                                           precomputed_file_transformation=precomputed_file_transformation,
                                           runtime_transformation=runtime_transformation)
        val_dataset = voc.TransformedVOC(root=cfg['dataset_path'], split='seg11valid',
                                         precomputed_file_transformation=precomputed_file_transformation,
                                         runtime_transformation=runtime_transformation)
    elif dataset_type == 'cityscapes':
        train_dataset = cityscapes.TransformedCityscapes(root=cfg['dataset_path'], split='train',
                                                         precomputed_file_transformation=precomputed_file_transformation,
                                                         runtime_transformation=runtime_transformation)
        val_dataset = cityscapes.TransformedCityscapes(root=cfg['dataset_path'], split='val',
                                                       precomputed_file_transformation=precomputed_file_transformation,
                                                       runtime_transformation=runtime_transformation)
    elif dataset_type == 'synthetic':
        if isinstance(precomputed_file_transformation,
                      dataset_precomputed_file_transformations.InstanceOrderingPrecomputedDatasetFileTransformation):
            precomputed_file_transformation = None  # Remove it, because we're going to order them when generating
            # the images instead.
        if precomputed_file_transformation is not None:
            raise ValueError('Cannot perform file transformations on the synthetic dataset.')
        train_dataset = synthetic.TransformedInstanceDataset(
            raw_dataset=synthetic.BlobExampleGenerator(n_images=pop_without_del(cfg, 'n_images_train', None),
                                                       ordering=cfg['ordering'],
                                                       intermediate_write_path=cfg['dataset_path']),
            raw_dataset_returns_images=True,
            runtime_transformation=runtime_transformation)
        val_dataset = synthetic.TransformedInstanceDataset(
            raw_dataset=synthetic.BlobExampleGenerator(n_images=pop_without_del(cfg, 'n_images_train', None),
                                                       ordering=cfg['ordering'],
                                                       intermediate_write_path=cfg['dataset_path']),
            raw_dataset_returns_images=True,
            runtime_transformation=runtime_transformation)
    else:
        if dataset_type in REGISTERED_DATASET_TYPES:
            raise NotImplementedError('Dataset type {} is registered, but I don\'t know how to instantiate it.'.format(
                dataset_type))
        else:
            raise NotImplementedError('I don\'t know of dataset type {}.  It\'s not registered.'.format(dataset_type))

    if not transform:
        for dataset in [train_dataset, val_dataset]:
            dataset.should_use_precompute_transform = False
            dataset.should_use_runtime_transform = False

    return train_dataset, val_dataset


def get_dataloaders(cfg, dataset_type, cuda, sampler_cfg=None):
    # 1. dataset
    train_dataset, val_dataset = get_datasets_with_transformations(dataset_type, cfg, transform=True)

    # 2. samplers
    if sampler_cfg is None:
        train_sampler = samplers.sampler.RandomSampler(train_dataset)
        val_sampler = samplers.sampler.SequentialSampler(val_dataset)
        train_for_val_sampler = samplers.sampler.SequentialSampler(train_dataset)
    else:
        train_sampler_cfg = sampler_cfg['train']
        val_sampler_cfg = sampler_cfg['val']
        train_for_val_cfg = pop_without_del(sampler_cfg, 'train_for_val', None)
        sampler_cfg['train_for_val'] = train_for_val_cfg

        sem_cls_filter = pop_without_del(train_sampler_cfg, 'sem_cls_filter', None)
        if sem_cls_filter is not None:
            if isinstance(sem_cls_filter[0], str):
                raw_train_dataset = train_dataset.raw_dataset if cfg['semantic_subset'] is not None \
                    else train_dataset
                try:
                    sem_cls_filter = [raw_train_dataset.semantic_class_names.index(class_name)
                                      for class_name in sem_cls_filter]
                except:
                    sem_cls_filter = [int(np.where(raw_train_dataset.semantic_class_names == class_name)[0][0])
                                      for class_name in sem_cls_filter]
        if dataset_type == 'voc':
            dataset_path = voc.get_default_voc_root()
        elif dataset_type == 'cityscapes':
            dataset_path = cityscapes.get_default_cityscapes_root()
        elif dataset_type == 'synthetic':
            dataset_path = '/tmp/'
        else:
            raise Exception('dataset_path not set for {}')
        train_instance_count_file = os.path.join(dataset_path, 'train_instance_counts.npy')
        train_sampler = get_configured_sampler(dataset_type, train_dataset, sequential=True,
                                               n_instances_range=pop_without_del(train_sampler_cfg,
                                                                                 'n_instances_range', None),
                                               n_images=pop_without_del(train_sampler_cfg, 'n_images', None),
                                               sem_cls_filter=sem_cls_filter,
                                               instance_count_file=train_instance_count_file)
        if isinstance(val_sampler_cfg, str) and val_sampler_cfg == 'copy_train':
            val_sampler = train_sampler.copy(sequential_override=True)
            val_dataset = train_dataset
        else:
            sem_cls_filter = pop_without_del(val_sampler_cfg, 'sem_cls_filter', None)
            if sem_cls_filter is not None:
                raw_val_dataset = val_dataset.raw_dataset if cfg['semantic_subset'] is not None \
                    else val_dataset
                if isinstance(sem_cls_filter[0], str):
                    try:
                        sem_cls_filter = [raw_val_dataset.semantic_class_names.index(class_name)
                                          for class_name in sem_cls_filter]
                    except:
                        sem_cls_filter = [int(np.where(raw_val_dataset.semantic_class_names == class_name)[0][0])
                                          for class_name in sem_cls_filter]
            val_instance_count_file = os.path.join(voc.VOC_ROOT, 'val_instance_counts.npy')
            val_sampler = get_configured_sampler(dataset_type, val_dataset, sequential=True,
                                                 n_instances_range=pop_without_del(val_sampler_cfg,
                                                                                   'n_instances_range', None),
                                                 n_images=pop_without_del(val_sampler_cfg, 'n_images', None),
                                                 sem_cls_filter=sem_cls_filter,
                                                 instance_count_file=val_instance_count_file)

        cut_n_images = pop_without_del(train_for_val_cfg, 'n_images', None) or len(train_dataset)
        train_for_val_sampler = train_sampler.copy(sequential_override=True,
                                                   cut_n_images=None if cut_n_images is None
                                                   else min(cut_n_images, len(train_sampler)))

    # Create dataloaders from datasets and samplers
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, sampler=train_sampler, **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, sampler=val_sampler, **loader_kwargs)
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset, batch_size=1,
                                                       sampler=train_for_val_sampler, **loader_kwargs)

    if DEBUG_ASSERTS:
        try:
            i, [sl, il] = [d for i, d in enumerate(train_loader) if i == 0][0]
        except:
            raise

    return {
        'train': train_loader,
        'val': val_loader,
        'train_for_val': train_loader_for_val,
    }


def pop_without_del(dictionary, key, default):
    val = dictionary.pop(key, default)
    dictionary[key] = val
    return val
