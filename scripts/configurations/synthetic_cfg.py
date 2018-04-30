from . import generic_cfg

default_config = generic_cfg.default_config
default_config.update(
    max_iteration=10000,
    interval_validate=100,
    lr=1.0e-7,
    size_average=True,
    n_instances_per_class=None,
)

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(),
    1: dict(  # 'semantic': mapping all semantic into a single instance
        max_iteration=10000,
        interval_validate=100,
        single_instance=True,
        lr=1.0e-10
    ),
    2: dict(  # instance seg. with initialization from semantic
        max_iteration=10000,
        interval_validate=100,
        initialize_from_semantic=True,
        bottleneck_channel_capacity='semantic',
        size_average=False,
        score_multiplier=0.00001,
    ),
    3: dict(  # instance seg. with S channels in the bottleneck layers
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        bottleneck_channel_capacity='semantic',
        size_average=False
    ),
    4: dict(  # instance seg. with semantic init. and N channels in the bottleneck layers
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        bottleneck_channel_capacity=None,
        initialize_from_semantic=True,
        size_average=False
    ),
    5: dict(  # instance seg. with an extra instance channel and semantic init.
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        bottleneck_channel_capacity='semantic',
        initialize_from_semantic=True,
        n_instances_per_class=3,
        size_average=False
    ),
    6: dict(  # instance seg. with initialization from semantic
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        initialize_from_semantic=False,
        bottleneck_channel_capacity='semantic',
        size_average=False,
        score_multiplier=1.0,
    ),
    7: dict(  # instance seg. WITHOUT initialization from semantic
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        initialize_from_semantic=False,
        bottleneck_channel_capacity=None,
        size_average=False,
        score_multiplier=None,
    ),
    8: dict(  # instance seg. with initialization from semantic
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        initialize_from_semantic=True,
        bottleneck_channel_capacity=None,
        size_average=False,
        score_multiplier=None,
    ),
    9: dict(  # 'semantic': mapping all semantic into a single instance
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        single_instance=True,
        size_average=False,
        bottleneck_channel_capacity=7,
    ),
    10: dict(  # 'semantic': mapping all semantic into a single instance
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-10,
        single_instance=False,
        size_average=False,
        n_instances_per_class=None,
        score_multiplier=None,
        initialize_from_semantic=False,
    ),
}
