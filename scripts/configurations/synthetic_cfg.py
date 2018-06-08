from . import generic_cfg


def get_default_config():
    _default_config = generic_cfg.get_default_config()
    _default_config.update(
        max_iteration=10000,
        interval_validate=100,
        lr=1.0e-4,
        size_average=True,
        n_instances_per_class=3,
        synthetic_generator_n_instances_per_semantic_id=2,
        export_activations=False,
        dataset='synthetic',
        n_images_train=100,
        n_images_val=100,
    )
    return _default_config


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(),
}
