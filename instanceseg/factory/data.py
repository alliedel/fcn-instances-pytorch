import torch
import torch.utils.data

from instanceseg.datasets import dataset_generator_registry
from instanceseg.factory import samplers as sampler_factory

DEBUG_ASSERTS = True


def get_datasets_with_transformations(dataset_type, cfg, transform=True):
    train_dataset, val_dataset = dataset_generator_registry.get_dataset(dataset_type, cfg,
                                                                        transform)
    return train_dataset, val_dataset


def get_dataloaders(cfg, dataset_type, cuda, sampler_cfg=None):
    # 1. dataset
    train_dataset, val_dataset = get_datasets_with_transformations(dataset_type, cfg,
                                                                   transform=True)

    # 2. samplers
    if sampler_cfg is not None and isinstance(sampler_cfg['val'], str) and \
            sampler_cfg['val'] == 'copy_train':
        val_dataset = train_dataset
    train_sampler, val_sampler, train_for_val_sampler = sampler_factory.get_samplers(
        dataset_type, sampler_cfg, train_dataset, val_dataset)

    # Create dataloaders from datasets and samplers
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['train_batch_size'],
                                               sampler=train_sampler,
                                               **loader_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg['val_batch_size'],
                                             sampler=val_sampler, **loader_kwargs)
    train_loader_for_val = torch.utils.data.DataLoader(train_dataset, batch_size=cfg[
        'val_batch_size'], sampler=train_for_val_sampler, **loader_kwargs)

    if DEBUG_ASSERTS:
        #        try:
        #            i, [sl, il] = [d for i, d in enumerate(train_loader) if i == 0][0]
        #        except:
        #            raise
        pass
    return {
        'train': train_loader,
        'val': val_loader,
        'train_for_val': train_loader_for_val,
    }


