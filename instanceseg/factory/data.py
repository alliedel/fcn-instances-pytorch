import os

import numpy as np
import torch

from instanceseg.datasets import dataset_generator_registry, sampler, dataset_registry
from instanceseg.utils.misc import pop_without_del
from instanceseg.factory.samplers import get_configured_sampler


DEBUG_ASSERTS = True


def get_datasets_with_transformations(dataset_type, cfg, transform=True):
    train_dataset, val_dataset = dataset_generator_registry.get_dataset(dataset_type, cfg, transform)
    return train_dataset, val_dataset


def get_dataloaders(cfg, dataset_type, cuda, sampler_cfg=None):
    # 1. dataset
    train_dataset, val_dataset = get_datasets_with_transformations(dataset_type, cfg, transform=True)

    # 2. samplers
    train_sampler, val_sampler, train_for_val_sampler = get_samplers(dataset_type, sampler_cfg,
                                                                     train_dataset, val_dataset)
    if sampler_cfg is not None and isinstance(sampler_cfg['val'], str) and sampler_cfg['val'] == 'copy_train':
        val_dataset = train_dataset

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


def get_samplers(dataset_type, sampler_cfg, train_dataset, val_dataset):

    if sampler_cfg is None:
        train_sampler = sampler.sampler.RandomSampler(train_dataset)
        val_sampler = sampler.sampler.SequentialSampler(val_dataset)
        train_for_val_sampler = sampler.sampler.SequentialSampler(train_dataset)
    else:
        # Get 'clean' datasets for instance counting
        default_train_dataset, default_val_dataset, transformer_tag = \
            dataset_generator_registry.get_default_datasets_for_instance_counts(dataset_type)

        # train sampler
        train_sampler = sampler_generator_helper(dataset_type, train_dataset, default_train_dataset,
                                                 sampler_cfg, 'train', transformer_tag)

        # val sampler
        if isinstance(sampler_cfg['val'], str) and sampler_cfg['val'] == 'copy_train':
            val_sampler = train_sampler.copy(sequential_override=True)
        else:
            val_sampler = sampler_generator_helper(dataset_type, val_dataset, default_val_dataset,
                                                   sampler_cfg, 'val', transformer_tag)

        # train_for_val sampler
        sampler_cfg['train_for_val'] = pop_without_del(sampler_cfg, 'train_for_val', None)
        cut_n_images = sampler_cfg['train_for_val'].n_images or len(train_dataset)
        train_for_val_sampler = train_sampler.copy(sequential_override=True,
                                                   cut_n_images=None if cut_n_images is None
                                                   else min(cut_n_images, len(train_sampler)))

    return train_sampler, val_sampler, train_for_val_sampler


def sampler_generator_helper(dataset_type, dataset, default_dataset, sampler_cfg, sampler_type, transformer_tag):
    instance_count_file = os.path.join(dataset_registry.REGISTRY[dataset_type].dataset_path,
                                       '{}_instance_counts_{}.npy'.format(sampler_type, transformer_tag))
    sem_cls_filter = sampler_cfg[sampler_type].sem_cls_filter
    sem_cls_filter_values = convert_sem_cls_filter_from_names_to_values(
        sem_cls_filter, default_dataset.semantic_class_names) \
        if sem_cls_filter is not None and isinstance(sem_cls_filter[0], str) else sem_cls_filter
    sampler = get_configured_sampler(
        dataset, default_dataset, sequential=True,
        n_instances_range=sampler_cfg[sampler_type].n_instances_range,
        n_images=sampler_cfg[sampler_type].n_images,
        sem_cls_filter=sem_cls_filter_values, instance_count_file=instance_count_file)
    return sampler


def convert_sem_cls_filter_from_names_to_values(sem_cls_filter, semantic_class_names):
    try:
        sem_cls_filter_values = [semantic_class_names.index(class_name)
                                 for class_name in sem_cls_filter]
    except:
        sem_cls_filter_values = [int(np.where(semantic_class_names == class_name)[0][0])
                                 for class_name in sem_cls_filter]
    return sem_cls_filter_values