from . import generic_cfg
from torchfcn.datasets import cityscapes


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
    'person_noaug_sem': dict(
        semantic_subset=['person', 'background'],
        interval_validate=1000,
        max_iteration=100000,
        n_instances_per_class=3,
        freeze_vgg=False,
        augment_semantic=False,
        map_to_semantic=True
    ),
    'semantic': dict(
        n_instances_per_class=1,
        max_iteration=1000000,
        single_instance=True,
        interval_validate=4000,
        semantic_subset=None
    )
}
