import os

import numpy as np
import torch

from torchfcn.datasets import samplers, dataset_utils, voc, \
    dataset_precomputed_file_transformations, dataset_runtime_transformations
from torchfcn.script_utils import DEBUG_ASSERTS
from torchfcn.utils.samplers import get_configured_sampler


def get_synthetic_datasets(cfg, transform=True):
    dataset_kwargs = dict(transform=transform, n_max_per_class=cfg['synthetic_generator_n_instances_per_semantic_id'],
                          map_to_single_instance_problem=cfg['single_instance'], ordering=cfg['ordering'],
                          semantic_subset=cfg['semantic_subset'])
    train_dataset = get_dataset_with_transformations(
        dataset_type='synthetic', split='train', **dataset_kwargs, n_images=cfg.pop('n_images_train', None))
    val_dataset = get_dataset_with_transformations(
        **dataset_kwargs, n_images=cfg.pop('n_images_val', None))

    # train_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs, n_images=cfg.pop(
    #     'n_images_train', None))
    # val_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs, n_images=cfg.pop(
    #     'n_images_val', None))
    return train_dataset, val_dataset


def get_voc_datasets(cfg, transform=True):
    dataset_kwargs = dict(transform=transform,
                          map_to_single_instance_problem=cfg['single_instance'],
                          ordering=cfg['ordering'], semantic_subset=cfg['semantic_subset'],
                          n_inst_cap_per_class=cfg['dataset_instance_cap'])
    train_dataset_kwargs = dict()
    train_dataset = get_dataset_with_transformations(
        dataset_type='voc', split='train', **dataset_kwargs, **train_dataset_kwargs)
    val_dataset = get_dataset_with_transformations(
        dataset_type='voc', split='seg11valid', **dataset_kwargs)
    return train_dataset, val_dataset


def get_cityscapes_datasets(cfg, transform=True):
    dataset_kwargs = dict(
        transform=transform,
        # map_to_single_instance_problem=cfg['single_instance'],
        # ordering=cfg['ordering'], semantic_subset=cfg['semantic_subset'],
        # n_inst_cap_per_class=cfg['dataset_instance_cap']
    )
    train_dataset_kwargs = dict()
    train_dataset = get_dataset_with_transformations(
        dataset_type='cityscapes', root=cityscapes_root, split='train', **dataset_kwargs, **train_dataset_kwargs)
    val_dataset = get_dataset_with_transformations(
        dataset_type='cityscapes', root=cityscapes_root, split='seg11valid', **dataset_kwargs)
    return train_dataset, val_dataset


def get_dataloaders(cfg, dataset_type, cuda, sampler_cfg=None):
    # 1. dataset
    if dataset_type == 'synthetic':
        train_dataset, val_dataset = get_synthetic_datasets(cfg)
    elif dataset_type == 'voc':
        train_dataset, val_dataset = get_voc_datasets(cfg)
    elif dataset_type == 'cityscapes':
        train_dataset, val_dataset = get_cityscapes_datasets(cfg)
    else:
        raise ValueError('dataset_type={} not recognized'.format(dataset_type))

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
        train_instance_count_file = os.path.join(VOC_ROOT, 'train_instance_counts.npy')
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
            val_instance_count_file = os.path.join(VOC_ROOT, 'val_instance_counts.npy')
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


def get_dataset_with_transformations(dataset_type, split, transform=True, resize=None, resize_size=None,
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
        dataset = voc.TransformedVOC(root=voc.VOC_ROOT, split=split,
                                     precomputed_file_transformation=precomputed_file_transformation,
                                     runtime_transformation=runtime_transformation)
    else:
        raise NotImplementedError('I don\'t know dataset of type {}'.format(dataset_type))

    if not transform:
        dataset.should_use_precompute_transform = False
        dataset.should_use_runtime_transform = False

    return dataset
