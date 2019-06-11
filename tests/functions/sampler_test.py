import os

import torch.utils.data
import tqdm
import shutil

from instanceseg.datasets import dataset_generator_registry
from scripts.configurations import cityscapes_cfg
from instanceseg.datasets import sampler
from instanceseg.factory import samplers as sampler_factory
from scripts.configurations import sampler_cfg_registry
from instanceseg.train.trainer_exporter import TrainerExporter
from instanceseg.analysis.visualization_utils import label2rgb, write_image

testing_level = 0


def test_instance_sampler(train_dataset, val_dataset, loader_kwargs):
    # Get sampler
    sampler_cfg = sampler_cfg_registry.sampler_cfgs['instance_test']
    train_sampler, val_sampler, train_for_val_sampler = sampler_factory.get_samplers('cityscapes', sampler_cfg,
                                                                                     train_dataset, val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                               sampler=train_sampler, **loader_kwargs)
    class_vals_present = [train_dataset.semantic_class_names.index(cls_name)
                          for cls_name in sampler_cfg['train'].sem_cls_filter_names]
    cfg_n_instances_per_class = sampler_cfg['train'].n_instances_ranges
    for idx, (d, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(train_loader),
                                                   desc='Iterating through new dataloader',
                                                   total=len(train_loader)):
        for sem_val, n_inst_range in zip(class_vals_present, cfg_n_instances_per_class):
            if n_inst_range is not None and n_inst_range[0] is not None:  # min number of instances of this class
                num_pixels_this_cls = (sem_lbl == sem_val).sum()
                assert num_pixels_this_cls > 0  # Assert that the class exists
                n_instances_this_cls = len(torch.unique(inst_lbl[sem_lbl == sem_val]))
                assert n_instances_this_cls >= n_inst_range[0]
                if n_inst_range[1] is not None:
                    assert n_instances_this_cls <= n_inst_range[1]


def y_or_n_input(msg_to_user):
    y_or_n = input(msg_to_user)
    while y_or_n not in ['y', 'n']:
        print('Answer y or n.')
        y_or_n = input(msg_to_user)
    return y_or_n


def write_images_and_confirm(dataloader, rule_as_string_to_user):
    img_dir = '/tmp/unittest/'
    if os.path.exists(img_dir):
        y_or_n = y_or_n_input('{} exists.  Would you like to remove it? (y/n)'.format(img_dir))
        if y_or_n == 'y':
            shutil.rmtree(img_dir)
        else:
            msg = 'Specify a new directory:'
            new_dir = input(msg)
            while new_dir != '' and os.path.exists(new_dir):
                new_dir = input('Directory already exists. \n' + msg)
    os.makedirs(img_dir)

    for idx, (d, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(dataloader),
                                                   desc='Iterating through new dataloader',
                                                   total=len(dataloader)):
        batch_sz = sem_lbl.size(0)
        sem_lbl_rgb = label2rgb(sem_lbl.numpy())
        inst_lbl_rgb = label2rgb(inst_lbl.numpy())
        for img_idx in range(batch_sz):
            img_untransformed, _ = \
                TrainerExporter.untransform_data(dataloader, d[img_idx, ...], None)
            write_image(os.path.join(img_dir, 'inst_lbl_{:06d}.png'.format(idx)), inst_lbl_rgb[
                img_idx, ...])
            write_image(os.path.join(img_dir, 'sem_lbl_{:06d}.png'.format(idx)), sem_lbl_rgb[
                img_idx, ...])
            write_image(os.path.join(img_dir, 'img_{:06d}.png'.format(idx)), img_untransformed)

    msg_to_user = 'Confirm that the images written to {} follow the rule: ' \
                  '{} y/n:'.format(img_dir, rule_as_string_to_user)
    y_or_n = input(msg_to_user)
    if y_or_n == 'n':
        raise Exception('Test error according to user')

    y_or_n = y_or_n_input('Remove test directory {}?'.format(img_dir))
    if y_or_n == 'y':
        shutil.rmtree(img_dir)


def test_all_cityscapes_occlusion_sampler(train_dataset, val_dataset, loader_kwargs):
    sampler_cfg = sampler_cfg_registry.sampler_cfgs['occlusion_more_than_1']
    assert sampler_cfg['train'].n_occlusions_range is not None
    train_sampler, val_sampler, train_for_val_sampler = sampler_factory.get_samplers(
        'cityscapes', sampler_cfg, train_dataset, val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                               sampler=train_sampler, **loader_kwargs)
    n_occlusions_range = sampler_cfg['train'].n_occlusions_range
    write_images_and_confirm(train_loader, 'Number of occlusions in image is >={} and '
                                           '<{}'.format(n_occlusions_range[0],
                                                        n_occlusions_range[1]))


def test_occlusion_sampler(train_dataset, val_dataset, loader_kwargs):
    # Get sampler
    sampler_cfg = sampler_cfg_registry.sampler_cfgs['occlusion_test']
    assert sampler_cfg['train'].n_occlusions_range is not None
    train_sampler, val_sampler, train_for_val_sampler = sampler_factory.get_samplers(
        'cityscapes', sampler_cfg, train_dataset, val_dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False,
                                               sampler=train_sampler, **loader_kwargs)
    n_occlusions_range = sampler_cfg['train'].n_occlusions_range
    write_images_and_confirm(train_loader, 'Number of occlusions in image is >={} and '
                                           '<{}'.format(n_occlusions_range[0],
                                                        n_occlusions_range[1]))


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
    print('Running occlusion on all cityscapes')
    test_all_cityscapes_occlusion_sampler(train_dataset, val_dataset, loader_kwargs)
    print('Running occlusion-based sampler test')
    test_occlusion_sampler(train_dataset, val_dataset, loader_kwargs)
    print('Running instance-based sampler test')
    test_instance_sampler(train_dataset, val_dataset, loader_kwargs)
    print('Running single-image test')
    test_single_image_sampler(train_dataset, loader_kwargs, image_index=10)
    print('Running vanilla test')
    test_vanilla_sampler(train_dataset, loader_kwargs)


if __name__ == '__main__':
    main()
