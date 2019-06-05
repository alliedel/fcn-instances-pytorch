import os

import torch.utils.data
import tqdm

from instanceseg.datasets import dataset_generator_registry
from scripts.configurations import cityscapes_cfg
from instanceseg.datasets import sampler
from instanceseg.factory import samplers as sampler_factory
from scripts.configurations import sampler_cfg_registry

testing_level = 0


def test_instance_sampler(train_dataset, val_dataset, loader_kwargs):
    # Get sampler
    sampler_cfg = sampler_cfg_registry.sampler_cfgs['instance_test']
    train_sampler, val_sampler, train_for_val_sampler = sampler_factory.get_samplers('cityscapes', sampler_cfg,
                                                                                     train_dataset, val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                               sampler=train_sampler, **loader_kwargs)
    class_vals_present = [train_dataset.semantic_class_names.index(cls_name)
                          for cls_name in sampler_cfg['train'].sem_cls_filter]
    cfg_n_instances_per_class = sampler_cfg['train'].n_instances_ranges
    for idx, (d, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(train_loader), desc='Iterating through new dataloader',
                                                   total=len(train_loader)):
        for sem_val, n_inst_range in zip(class_vals_present, cfg_n_instances_per_class):
            if n_inst_range is not None and n_inst_range[0] is not None:  # min number of instances of this class
                num_pixels_this_cls = (sem_lbl == sem_val).sum()
                assert num_pixels_this_cls > 0  # Assert that the class exists
                n_instances_this_cls = torch.unique(inst_lbl[sem_lbl == sem_val])
                assert n_instances_this_cls >= n_inst_range[0]
                if n_inst_range[1] is not None:
                    assert n_instances_this_cls <= n_inst_range[1]


def test_vanilla_sampler(train_dataset, loader_kwargs):
    # Get sampler
    full_sequential_train_sampler = sampler.get_pytorch_sampler(sequential=True)(train_dataset)
    full_random_train_sampler = sampler.get_pytorch_sampler(sequential=False)(train_dataset)

    # Apply sampler to dataloaders
    sequential_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                                          sampler=full_sequential_train_sampler, **loader_kwargs)
    shuffled_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                                        sampler=full_random_train_sampler, **loader_kwargs)
    for train_loader in [sequential_train_loader, shuffled_train_loader]:
        for idx, (d, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(train_loader), desc='Iterating through new dataloader',
                                                       total=len(train_loader)):
            pass
        assert len(train_loader) == len(train_dataset)

    # Make sure shuffling/not shuffling is correct
    for compare_i in range(10):  # don't bother checking the whole dataset for computational reasons.
        indexed_image_from_sequential_loader = [d for i, (d, ls) in enumerate(sequential_train_loader)
                                                if i == compare_i][0][0, ...]
        assert torch.equal(train_dataset[compare_i][0], indexed_image_from_sequential_loader)
    subset_sequential_train_loader = [d for i, d in enumerate(sequential_train_loader) if i < 10]
    subset_random_train_loader = [d for i, d in enumerate(shuffled_train_loader) if i < 10]
    assert not all([torch.equal(data1[0], data2[0]) for (data1, data2) in zip(subset_random_train_loader,
                                                                              subset_sequential_train_loader)])


def test_single_image_sampler(train_dataset, loader_kwargs, image_index=0):
    # Get sampler
    bool_index_subset = [idx == image_index for idx in range(len(train_dataset))]
    single_image_train_sampler = sampler.get_pytorch_sampler(sequential=True,
                                                             bool_index_subset=bool_index_subset)(train_dataset)
    shuffled_single_image_train_sampler = sampler.get_pytorch_sampler(sequential=False,
                                                                      bool_index_subset=bool_index_subset)(train_dataset)

    # Apply sampler to dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                               sampler=single_image_train_sampler, **loader_kwargs)
    train_loader_shuffled = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                                        sampler=shuffled_single_image_train_sampler, **loader_kwargs)

    for idx, (d, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(train_loader), desc='Iterating through new dataloader',
                                                   total=len(train_loader)):
        pass
    assert len(train_loader) == 1
    image_from_train_loader = [d for d in train_loader][0]
    assert all([torch.equal(data1[0], data2[0][0, ...]) for data1, data2 in zip(train_dataset[image_index],
                                                                                image_from_train_loader)])
    assert all([torch.equal(data1[0], data2[0]) for (data1, data2) in zip(train_loader,
                                                                          train_loader_shuffled)])


def main():
    # Setup
    cfg = cityscapes_cfg.get_default_config()
    print('Getting datasets')
    train_dataset, val_dataset = dataset_generator_registry.get_dataset('cityscapes', cfg)
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    print('Running instance-based sampler test')
    test_instance_sampler(train_dataset, val_dataset, loader_kwargs)
    print('Running single-image test')
    test_single_image_sampler(train_dataset, loader_kwargs, image_index=10)
    print('Running vanilla test')
    test_vanilla_sampler(train_dataset, loader_kwargs)


if __name__ == '__main__':
    main()
