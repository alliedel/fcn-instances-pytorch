from . import generic_cfg


def get_default_config():
    default_cfg = generic_cfg.default_config
    default_cfg.update(
        dict(n_instances_per_class=3,
             set_extras_to_void=True,
             lr=1.0e-4,
             size_average=True,
             )
    )
    return default_cfg


default_config = get_default_config()


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(),
    1: dict(
        n_instances_per_class=1,
        set_extras_to_void=False
    ),
    2: dict(
        semantic_only_labels=True,
        n_instances_per_class=1,
        set_extras_to_void=False
    ),
    3: dict(
        semantic_only_labels=False,
        n_instances_per_class=3,
        set_extras_to_void=True
    ),
    4: dict(
        semantic_subset=['person', 'background'],
        set_extras_to_void=True,
    ),
    5: dict(
        semantic_only_labels=False,
        n_instances_per_class=3,
        set_extras_to_void=True,
        max_iteration=1000000
    ),
    6: dict(  # semantic, single-instance problem
        n_instances_per_class=None,
        set_extras_to_void=True,
        max_iteration=1000000,
        single_instance=True
    ),
    7: dict(
        semantic_only_labels=False,
        n_instances_per_class=3,
        set_extras_to_void=True,
        weight_by_instance=True
    ),
    8: dict(
        n_instances_per_class=3,
        set_extras_to_void=True,
        weight_by_instance=True,
        semantic_subset=['person', 'background'],
    ),
    9: dict(  # created to reduce memory
        n_instances_per_class=3,
        semantic_subset=['person', 'car', 'background'],
    ),
    10: dict(
        n_instances_per_class=3,
        semantic_subset=['person', 'car', 'background'],
        lr=1e-6
    ),
    11: dict(
        semantic_subset=['person', 'background'],
        set_extras_to_void=True,
        interval_validate=4000,
        max_iteration=10000000,
    ),
    'person_only__freeze_vgg__many_itr': dict(
        semantic_subset=['person', 'background'],
        set_extras_to_void=True,
        interval_validate=100,
        max_iteration=10000,
    ),
    'person_only__3_channels_map_to_semantic__freeze_vgg__many_itr': dict(
        semantic_subset=['person', 'background'],
        set_extras_to_void=True,
        interval_validate=4000,
        max_iteration=100000,
        map_to_semantic=True,
        n_instances_per_class=3,
    )
}
