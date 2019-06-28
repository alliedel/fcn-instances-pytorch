from typing import Type, Dict, Union

import os
import logging

from torch.utils.data import SequentialSampler, RandomSampler

from instanceseg.datasets import sampler, dataset_statistics, indexfilter, dataset_registry, \
    dataset_generator_registry
from instanceseg.utils.misc import pop_without_del

logger = logging.getLogger(__name__)


def get_valid_indices_given_dataset(dataset_configured_for_stats,
                                    sampler_config_with_vals: sampler.SamplerConfig,
                                    instance_count_file=None, semantic_pixel_count_file=None,
                                    occlusion_counts_file=None):
    semantic_class_pixel_counts_cache = dataset_statistics.PixelsPerSemanticClass(
        range(len(dataset_configured_for_stats.semantic_class_names)),
        cache_file=semantic_pixel_count_file)
    instance_counts_cache = dataset_statistics.NumberofInstancesPerSemanticClass(
        range(len(dataset_configured_for_stats.semantic_class_names)),
        cache_file=instance_count_file)
    occlusion_counts_cache = dataset_statistics.OcclusionsOfSameClass(range(len(
        dataset_configured_for_stats.semantic_class_names)), cache_file=occlusion_counts_file)

    if sampler_config_with_vals.requires_instance_counts:
        instance_counts_cache.compute_or_retrieve(dataset_configured_for_stats)
        instance_counts = instance_counts_cache.stat_tensor
    else:
        instance_counts = None
    if sampler_config_with_vals.requires_semantic_pixel_counts:
        semantic_class_pixel_counts_cache.compute_or_retrieve(dataset_configured_for_stats)
        semantic_pixel_counts = semantic_class_pixel_counts_cache.stat_tensor
    else:
        semantic_pixel_counts = None
    if sampler_config_with_vals.requires_occlusion_counts:
        occlusion_counts_cache.compute_or_retrieve(dataset_configured_for_stats)
        occlusion_counts = occlusion_counts_cache.stat_tensor
    else:
        occlusion_counts = None
    index_filter = indexfilter.ValidIndexFilter(n_original_images=len(dataset_configured_for_stats),
                                                sampler_config=sampler_config_with_vals,
                                                instance_counts=instance_counts,
                                                semantic_class_pixel_counts=semantic_pixel_counts,
                                                occlusion_counts=occlusion_counts)
    return index_filter.valid_indices


def get_configured_sampler(dataset, dataset_configured_for_stats, sequential,
                           sampler_config_with_vals, instance_count_file,
                           semantic_pixel_count_file, occlusion_counts_file):
    """
    Builds a sampler of a dataset, which requires a list of valid indices and whether it's random/sequential,
    as well as the dataset itself.

    dataset: the actual dataset you want to sample from in the end
    dataset_configured_for_stats: the dataset you want to compute stats from (to inform how you
        sample 'dataset') -- useful if you're going to get rid of semantic classes, etc. but want
        to still sample images that have them.
        If it matches dataset, just pass dataset in for this parameter as well.
    """
    if dataset_configured_for_stats is not None:
        assert len(dataset_configured_for_stats) == len(dataset)

    valid_indices = get_valid_indices_given_dataset(
        dataset_configured_for_stats, sampler_config_with_vals,
        instance_count_file=instance_count_file,
        semantic_pixel_count_file=semantic_pixel_count_file,
        occlusion_counts_file=occlusion_counts_file)

    my_sampler = sampler.get_pytorch_sampler(sequential, bool_index_subset=valid_indices)(dataset)
    if len(my_sampler) == 0:
        raise ValueError('length of sampler is 0; {} valid indices'.format(sum(valid_indices)))
    return my_sampler


def sampler_generator_helper(dataset_type, dataset, default_dataset, sampler_config, sampler_split_type,
                             transformer_tag):
    instance_count_filename = \
        dataset_registry.REGISTRY[dataset_type].get_instance_count_filename(sampler_split_type, transformer_tag)
    semantic_pixel_count_filename = dataset_registry.REGISTRY[dataset_type].get_semantic_pixel_count_filename(
        sampler_split_type, transformer_tag)
    occlusion_counts_filename = dataset_registry.REGISTRY[dataset_type].get_occlusion_counts_filename(
        sampler_split_type, transformer_tag)
    filter_config = sampler.SamplerConfig.create_from_cfg_without_vals(
        sampler_config[sampler_split_type], default_dataset.semantic_class_names)
    my_sampler = get_configured_sampler(
        dataset, default_dataset, sequential=True, sampler_config_with_vals=filter_config,
        instance_count_file=instance_count_filename,
        semantic_pixel_count_file=semantic_pixel_count_filename,
        occlusion_counts_file=occlusion_counts_filename)
    return my_sampler


def get_samplers(dataset_type, sampler_cfg, datasets, splits=('train', 'val', 'train_for_val')):
    samplers = {}
    if sampler_cfg is None:
        SamplerTypes = {
            'train': sampler.sampler.RandomSampler,
            'val': sampler.sampler.SequentialSampler,
            'train_for_val': sampler.sampler.SequentialSampler
        }
        for split in splits:
            samplers[split] = SamplerTypes[split](datasets[split])
    else:
        # Get 'clean' datasets for instance counting
        if not dataset_type == 'synthetic':
            default_datasets, transformer_tag = \
                dataset_generator_registry.get_default_datasets_for_instance_counts(dataset_type,
                                                                                    splits=('train', 'val'))
        else:
            default_datasets = datasets
            transformer_tag = 'no_transformation'
        # train sampler
        split = 'train'
        if split in splits:
            samplers[split] = sampler_generator_helper(dataset_type, datasets[split], default_datasets[split],
                                                       sampler_cfg, 'train', transformer_tag)

        # val sampler
        split = 'val'
        if split in splits:
            if isinstance(sampler_cfg[split], str) and sampler_cfg[split] == 'copy_train':
                assert 'train' in splits, 'train must be in split if val needs to copy sampler from train'
                samplers[split] = samplers['train'].copy(sequential_override=True)
            else:
                samplers[split] = sampler_generator_helper(dataset_type, datasets[split], default_datasets[split],
                                                           sampler_cfg, sampler_split_type=split,
                                                           transformer_tag=transformer_tag)

        # train_for_val sampler
        if 'train_for_val' in splits:
            assert 'train' in splits
            sampler_cfg['train_for_val'] = pop_without_del(sampler_cfg, 'train_for_val', None)
            cut_n_images = sampler_cfg['train_for_val'].n_images or len(datasets['train'])
            samplers['train_for_val'] = samplers['train'].copy(sequential_override=True,
                                                               cut_n_images=None if cut_n_images is None
                                                               else min(cut_n_images, len(samplers['train'])))

    return samplers
