# Writing a function to create list of instances and their sizes, given GT for one datapoint.


import os.path as osp
import sys

import torch

from scripts.configurations import sampler_cfg_registry
from instanceseg.datasets.dataset_generator_registry import get_default_datasets_for_instance_counts
from instanceseg.factory import samplers as sampler_factory
from instanceseg.utils.script_setup import setup_train

here = osp.dirname(osp.abspath(__file__))


def get_single_img_data(dataloader, idx=0):
    img, sem_lbl, inst_lbl = None, None, None
    for i, datapoint in enumerate(dataloader):
        if i == idx:
            img = datapoint['image']
            sem_lbl, inst_lbl = datapoint['sem_lbl'], datapoint['inst_lbl']

    return img, (sem_lbl, inst_lbl)


def extract_from_argv(argv, key, default_val=None):
    if key in argv:
        k_idx = argv.index(key)
        assert argv[k_idx + 1]
        val = argv[k_idx + 1]
        del argv[k_idx + 1]
        del argv[k_idx]
    else:
        val = default_val
    return val


def main():
    # args, cfg_override_args = local_setup(loss)
    argv = sys.argv[1:]
    gpu = extract_from_argv(argv, key='--gpu', default_val=None)
    if gpu is None:
        gpu = extract_from_argv(argv, key='-g', default_val=[0])
    config_idx = extract_from_argv(argv, key='-c', default_val='overfit_1')
    sampler_cfg = sampler_cfg_registry.get_sampler_cfg_set(sampler_arg=None)
    dataset_name = 'cityscapes'
    default_datasets, transformer_tag = get_default_datasets_for_instance_counts(dataset_name, ('train',))
    cuda = torch.cuda.is_available()
    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    batch_sizes = {split: 1 for split in default_datasets.keys()}
    samplers = sampler_factory.get_samplers(dataset_name, sampler_cfg, default_datasets,
                                            splits=default_datasets.keys())
    dataloaders = {
        split: torch.utils.data.DataLoader(default_datasets[split], batch_size=batch_sizes[split],
                                           sampler=samplers[split], **loader_kwargs)
        for split in default_datasets.keys()
    }

    img, (sem_lbl, inst_lbl) = get_single_img_data(dataloaders['train'], 0)

    print('Got an image!')


if __name__ == '__main__':
    main()
