from torchfcn.datasets import cityscapes
from . import generic_cfg


def get_default_config():
    default_cfg = generic_cfg.get_default_config()
    default_cfg.update(
        dict(n_instances_per_class=3,
             set_extras_to_void=True,
             lr=1.0e-4,
             dataset='cityscapes',
             resize=True,
             resize_size=(281, 500),  # (512, 1024)
             dataset_path=cityscapes.get_default_cityscapes_root()
             )
    )
    return default_cfg


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(),
    'person_only__semantic': dict(
        semantic_subset=['person', 'background'],
        set_extras_to_void=True,
        single_instance=True,
        n_instances_per_class=None,
    ),
    'car_only__single_instance_semantic': dict(
        semantic_subset=['car', 'background'],
        interval_validate=1000,
        max_iteration=100000,
        n_instances_per_class=1,
        single_instance=True,
        map_to_semantic=True,
        dataset_instance_cap=None,
    ),
    'car_only__mapped_semantic': dict(
        n_instances_per_class=1,
        max_iteration=1000000,
        single_instance=True,
        interval_validate=4000,
        semantic_subset=['car', 'background'],
        map_to_semantic=True,
        dataset_instance_cap=None,
    ),
    'semantic': dict(
        single_instance=True,
        n_instances_per_class=1,
        dataset_instance_cap=None,
    ),
    'car_instance': dict(
        max_iteration=1000000,
        interval_validate=4000,
        semantic_subset=None,
        dataset_instance_cap=3,
        n_instances_per_class=3,
        ordering='lr'
    )
}
