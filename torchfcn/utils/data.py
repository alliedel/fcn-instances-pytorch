import os

import numpy as np
import torch

import torchfcn
from torchfcn.datasets import samplers
from torchfcn.datasets.voc import VOC_ROOT
from torchfcn.script_utils import DEBUG_ASSERTS
from torchfcn.utils.samplers import get_configured_sampler


def get_synthetic_datasets(cfg, transform=True):
    dataset_kwargs = dict(transform=transform, n_max_per_class=cfg['synthetic_generator_n_instances_per_semantic_id'],
                          map_to_single_instance_problem=cfg['single_instance'], ordering=cfg['ordering'],
                          semantic_subset=cfg['semantic_subset'])
    train_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs, n_images=cfg.pop(
        'n_images_train', None))
    val_dataset = torchfcn.datasets.synthetic.BlobExampleGenerator(**dataset_kwargs, n_images=cfg.pop(
        'n_images_val', None))
    return train_dataset, val_dataset


def get_voc_datasets(cfg, voc_root, transform=True):
    dataset_kwargs = dict(transform=transform, semantic_only_labels=cfg['semantic_only_labels'],
                          set_extras_to_void=cfg['set_extras_to_void'],
                          map_to_single_instance_problem=cfg['single_instance'],
                          ordering=cfg['ordering'])
    train_dataset_kwargs = dict()
    train_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(voc_root, split='train', **dataset_kwargs,
                                                          **train_dataset_kwargs)
    val_dataset = torchfcn.datasets.voc.VOC2011ClassSeg(voc_root, split='seg11valid', **dataset_kwargs)
    return train_dataset, val_dataset


def get_dataloaders(cfg, dataset_type, cuda, sampler_cfg=None):
    # 1. dataset
    if dataset_type == 'synthetic':
        train_dataset, val_dataset = get_synthetic_datasets(cfg)
    elif dataset_type == 'voc':
        train_dataset, val_dataset = get_voc_datasets(cfg, VOC_ROOT)
        if cfg['semantic_subset'] is not None:
            train_dataset.reduce_to_semantic_subset(cfg['semantic_subset'])
            val_dataset.reduce_to_semantic_subset(cfg['semantic_subset'])
        instance_cap = cfg['n_instances_per_class'] if cfg['dataset_instance_cap'] == 'match_model' else \
            cfg['dataset_instance_cap']
        if instance_cap is not None:
            train_dataset.set_instance_cap(instance_cap)
            val_dataset.set_instance_cap(instance_cap)
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
                try:
                    sem_cls_filter = [train_dataset.class_names.index(class_name) for class_name in sem_cls_filter]
                except:
                    sem_cls_filter = [int(np.where(train_dataset.class_names == class_name)[0][0])
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
                if isinstance(sem_cls_filter[0], str):
                    try:
                        sem_cls_filter = [val_dataset.class_names.index(class_name) for class_name in sem_cls_filter]
                    except:
                        sem_cls_filter = [int(np.where(val_dataset.class_names == class_name)[0][0])
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