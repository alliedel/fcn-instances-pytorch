from . import generic_cfg

default_config = generic_cfg.default_config
default_config.update(
    max_iteration=10000,
    interval_validate=100,
    lr=1.0e-4,
    size_average=True,
    n_instances_per_class=None,
    synthetic_generator_n_instances_per_semantic_id=2
)

configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(),
}
