from . import generic_cfg


def get_default_config():
    _default_config = generic_cfg.get_default_config()
    _default_config.update(
        dataset_path='/tmp/',
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
        infinite_synthetic=False,
        one_dimension=None,  # {'x', 'y'}
        semantic_subset=None,
        img_size=None,
        portrait=False,
    )
    return _default_config


configurations = {
    # same configuration as original work
    # https://github.com/shelhamer/fcn.berkeleyvision.org
    0: dict(),
    'square2': dict(
        semantic_subset=['background', 'square'],
        synthetic_generator_n_instances_per_semantic_id=2,
        n_instances_per_class=2,
        infinite_synthetic=True
    ),
    'test_overfit_1_iou': dict(
        interval_validate=10,
        n_images_train=1,
        n_images_val=1,
        write_instance_metrics=True,
        export_activations=True,
        loss_type='soft_iou',
        size_average=False,
        lr=1.0e-3,
        max_iteration=10,
        sampler='overfit_1'
    ),
    'test_overfit_1_xent': dict(
        interval_validate=10,
        n_images_train=1,
        n_images_val=1,
        write_instance_metrics=True,
        export_activations=True,
        loss_type='cross_entropy',
        size_average=1,
        lr=1.0e-4,
        max_iteration=50,
        sampler='overfit_1'
    )
}
