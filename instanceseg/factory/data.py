import torch
import torch.utils.data

from instanceseg.datasets import dataset_generator_registry
from instanceseg.factory import samplers as sampler_factory

DEBUG_ASSERTS = True


def get_dataset_with_transformations(dataset_type, cfg, split, transform=True):
    dataset = dataset_generator_registry.get_dataset(dataset_type, cfg, split, transform)
    return dataset


def get_dataloaders(cfg, dataset_type, cuda, sampler_cfg=None, splits=('train', 'val', 'train_for_val')):
    non_derivative_splits = (s for s in splits if s != 'train_for_val')
    build_train_for_val = splits != non_derivative_splits

    # 1. dataset
    datasets = {
        split: dataset_generator_registry.get_dataset(dataset_type, cfg, split, transform=True)
                      for split in non_derivative_splits
    }
    if 'train_for_val' in splits:
        datasets['train_for_val'] = datasets['train']

    # 2. samplers
    if sampler_cfg is not None and 'val' in sampler_cfg and isinstance(sampler_cfg['val'], str) and \
            sampler_cfg['val'] == 'copy_train':
        assert 'train' in splits
        datasets['val'] = datasets['train']
    samplers = sampler_factory.get_samplers(dataset_type, sampler_cfg, datasets, splits=splits)

    # Create dataloaders from datasets and samplers
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    batch_sizes = {split: cfg['val_batch_size'] if split == 'train_for_val' else cfg['{}_batch_size'.format(split)]
                   for split in splits}
    dataloaders = {
        split: torch.utils.data.DataLoader(datasets[split], batch_size=batch_sizes[split],
                                           sampler=samplers[split], **loader_kwargs) for split in splits
    }
    if DEBUG_ASSERTS:
        #        try:
        #            i, [sl, il] = [d for i, d in enumerate(train_loader) if i == 0][0]
        #        except:
        #            raise
        pass
    return dataloaders