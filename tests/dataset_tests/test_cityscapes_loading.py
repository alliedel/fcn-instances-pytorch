import os

import numpy as np
import torch
import tqdm

import scripts.configurations.sampler_cfg
import instanceseg.factory.data
from instanceseg.datasets.dataset_registry import REGISTRY
from instanceseg.factory import samplers


def test_get_datasets():
    cfg = REGISTRY['cityscapes'].default_config
    cfg['ordering'] = None  # 'LR'
    train_dataset_, val_dataset_ = REGISTRY['cityscapes'].dataset_generator(cfg)
    img, (sem_lbl, inst_lbl) = train_dataset_.__getitem__(0)
    return train_dataset_, val_dataset_


def test_get_dataloaders_with_two_semantic_classes():
    cfg = REGISTRY['cityscapes'].default_config
    sampler_cfg = scripts.configurations.sampler_cfg.get_sampler_cfg()
    dataloaders = instanceseg.factory.data.get_dataloaders(cfg, 'cityscapes', cuda, sampler_cfg)
    person_val = dataloaders['train'].dataset.semantic_class_names.index('person')

    for img, (sem_lbl, inst_lbl) in dataloaders['train']:
        assert torch.sum(sem_lbl == person_val) > 0

    # Check that the unused images don't have people in them
    unused_indices = [idx for idx in dataloaders['train'].sampler.initial_indices
                      if idx not in dataloaders['train'].sampler.indices]
    unused_example = dataloaders['train'].dataset[unused_indices[1]]
    sem_lbl_unused = unused_example[1][1]
    assert torch.sum(sem_lbl_unused == person_val) == 0
    return dataloaders['train'], dataloaders['val']


def test_get_dataloaders_with_semantic_person_filtering():
    cfg = REGISTRY['cityscapes'].default_config
    sampler_cfg = scripts.configurations.sampler_cfg.get_sampler_cfg('person_2inst_20img_sameval')
    dataloaders = instanceseg.factory.data.get_dataloaders(cfg, 'cityscapes', cuda, sampler_cfg)
    person_val = dataloaders['train'].dataset.semantic_class_names.index('person')

    for img, (sem_lbl, inst_lbl) in dataloaders['train']:
        assert torch.sum(sem_lbl == person_val) > 0

    # Check that the unused images don't have people in them
    unused_indices = [idx for idx in dataloaders['train'].sampler.initial_indices
                      if idx not in dataloaders['train'].sampler.indices]
    unused_example = dataloaders['train'].dataset[unused_indices[1]]
    sem_lbl_unused = unused_example[1][1]
    assert torch.sum(sem_lbl_unused == person_val) == 0
    return dataloaders['train'], dataloaders['val']


def test_get_dataloaders_with_instance_car_mapping():
    cfg = REGISTRY['cityscapes'].default_config
    cfg['semantic_subset'] = ['car', 'background']
    cfg['dataset_instance_cap'] = None
    sampler_cfg = scripts.configurations.sampler_cfg.get_sampler_cfg('car_2_4')
    dataloaders_default = instanceseg.factory.data.get_dataloaders(cfg, 'cityscapes', cuda, sampler_cfg)
    cfg['dataset_instance_cap'] = 4
    dataloaders = instanceseg.factory.data.get_dataloaders(cfg, 'cityscapes', cuda, sampler_cfg)

    # Verify the image only has background and people, and that those locations line up with the initial dataset's
    # labels.
    first_index = dataloaders['val'].sampler.indices[7]
    img_ss, (sem_lbl_ss, inst_lbl_ss) = dataloaders['val'].dataset[first_index]
    img, (sem_lbl, inst_lbl) = dataloaders_default['val'].dataset[first_index]
    car_mask1 = sem_lbl == dataloaders_default['val'].dataset.semantic_class_names.index('car')
    car_mask2 = sem_lbl_ss == dataloaders['val'].dataset.semantic_class_names.index('car')

    assert np.all(car_mask1 == car_mask2) and car_mask1.sum() > 0

    import ipdb; ipdb.set_trace()

    return dataloaders['train'], dataloaders['val']


def test_loading_all(dataset):
    for idx, (img, (sem_lbl, inst_lbl)) in tqdm.tqdm(enumerate(dataset), desc='Loading Cityscape images',
                                                     total=len(dataset)):
        pass


if __name__ == '__main__':
    gpu = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    cuda = torch.cuda.is_available()

    # test1
    # train_dataset, val_dataset = test_get_datasets()
    train_dataloader, val_dataloader = test_get_dataloaders_with_instance_car_mapping()
