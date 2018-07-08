from . import generic_cfg


def get_default_config():
    default_cfg = generic_cfg.get_default_config()
    default_cfg.update(
        dict(n_instances_per_class=3,
             set_extras_to_void=True,
             lr=1.0e-4,
             dataset='cityscapes',
             resize=True,
             resize_size=(281, 500)  # (512, 1024)
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
}
