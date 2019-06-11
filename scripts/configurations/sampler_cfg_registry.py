# TODO(allie): Make it easier to generate and override sampler_cfgs
#  (like it is to generate the others)
from instanceseg.datasets.sampler import SamplerConfigWithoutValues


def get_sampler_cfg_set(n_images_train=None, n_images_val=None, n_images_train_for_val=None,
                        sem_cls_filter_names=None, n_instances_ranges=None,
                        n_occlusions_range=None, val_copy_train=False):
    sampler_cfg_kwargs = dict(n_images=n_images_train, sem_cls_filter_names=sem_cls_filter_names,
                              n_instances_ranges=n_instances_ranges,
                              n_occlusions_range=n_occlusions_range)
    assert val_copy_train is False or n_images_val is None
    train_sampler_cfg = SamplerConfigWithoutValues(**sampler_cfg_kwargs)
    val_sampler_cfg = SamplerConfigWithoutValues(**sampler_cfg_kwargs) if not val_copy_train \
        else train_sampler_cfg
    train_for_val_sampler_cfg = SamplerConfigWithoutValues(n_images=n_images_train_for_val)
    return {
        'train': train_sampler_cfg,
        'val': val_sampler_cfg,
        'train_for_val': train_for_val_sampler_cfg
    }


sampler_cfgs = {
    None: get_sampler_cfg_set(),
    'default': get_sampler_cfg_set(),
    'overfit_1_car_person': get_sampler_cfg_set(n_images_train=1, val_copy_train=True,
                                                sem_cls_filter_names=['car', 'person'],
                                                n_instances_ranges=[(2, 3 + 1), (2, 3 + 1)]),
    'overfit_10_car_person': get_sampler_cfg_set(n_images_train=10, val_copy_train=True,
                                                 sem_cls_filter_names=['car', 'person'],
                                                 n_instances_ranges=[(2, 3 + 1), (2, 3 + 1)]),
    'car_2_4': get_sampler_cfg_set(sem_cls_filter_names=['car'], n_instances_ranges=(2, 4 + 1)),
    'car_2_5': get_sampler_cfg_set(sem_cls_filter_names=['car'], n_instances_ranges=(2, 5 + 1)),
    'person_car_2_4': get_sampler_cfg_set(sem_cls_filter_names=['car', 'person'],
                                          n_instances_ranges=[(2, 4 + 1), (2, 4 + 1)]),
    'car_2_3': get_sampler_cfg_set(sem_cls_filter_names=['car'], n_instances_ranges=(2, 3 + 1)),
    'person_car_2_3': get_sampler_cfg_set(sem_cls_filter_names=['car', 'person'],
                                          n_instances_ranges=[(2, 3 + 1), (2, 3 + 1)]),
    'car_bus_train_1_3': get_sampler_cfg_set(sem_cls_filter_names=['car', 'bus', 'train'],
                                             n_instances_ranges=[(1, 3 + 1) for _ in range(3)]),
    'instance_test': get_sampler_cfg_set(sem_cls_filter_names=['person', 'background'],
                                         n_instances_ranges=[(1, 3 + 1), None], n_images_train=10,
                                         val_copy_train=True),
    'occlusion_test': get_sampler_cfg_set(sem_cls_filter_names=['car', 'background'],
                                          n_occlusions_range=(2, 5), n_images_train=10,
                                          val_copy_train=True),
    'occlusion_more_than_1':  get_sampler_cfg_set(sem_cls_filter_names=['car', 'background'],
                                                  n_occlusions_range=(2, 5),
                                                  val_copy_train=False)
}
sampler_cfgs['car_2_4inst_allimg_realval'] = sampler_cfgs['car_2_4']


def get_sampler_cfg(sampler_arg):
    sampler_cfg = sampler_cfgs[sampler_arg]
    if sampler_cfg['train_for_val'] is None:
        sampler_cfg['train_for_val'] = sampler_cfgs['default']['train_for_val']
    return sampler_cfg
