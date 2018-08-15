import os

import torch.utils.data
import tqdm

from instanceseg.datasets import dataset_generator_registry
from scripts.configurations import voc_cfg
from instanceseg.datasets import sampler


def test_vanilla_sampler(train_dataset, loader_kwargs):
    # Get sampler
    full_sequential_train_sampler = sampler.sampler_factory(sequential=True)(train_dataset)
    full_random_train_sampler = sampler.sampler_factory(sequential=False)(train_dataset)

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
    single_image_train_sampler = sampler.sampler_factory(sequential=True,
                                                         bool_index_subset=bool_index_subset)(train_dataset)
    shuffled_single_image_train_sampler = sampler.sampler_factory(sequential=False,
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
    cfg = voc_cfg.get_default_config()
    print('Getting datasets')
    train_dataset, val_dataset = dataset_generator_registry.get_dataset('voc', cfg)
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)

    loader_kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
    print('Running single-image test')
    test_single_image_sampler(train_dataset, loader_kwargs, image_index=10)
    print('Running vanilla test')
    test_vanilla_sampler(train_dataset, loader_kwargs)


if __name__ == '__main__':
    main()
