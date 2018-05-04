from . import generic_cfg

default_config = generic_cfg.default_config
default_config.update(
    dict(n_instances_per_class=3,
         set_extras_to_void=True,
         lr=1.0e-4,
         size_average=True,
         )
)

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
        filter_images_by_semantic_subset=True
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
        semantic_only_labels=False,
        n_instances_per_class=3,
        set_extras_to_void=True,
        weight_by_instance=True,
        semantic_subset=['person', 'background'],
        filter_by_semantic=True,
    ),
    9: dict(  # created to reduce memory
        n_instances_per_class=3,
        semantic_subset=['person', 'car', 'background'],
        filter_by_semantic=True
    )
}
