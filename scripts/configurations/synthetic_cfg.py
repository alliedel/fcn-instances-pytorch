from . import generic_cfg


class SYNTHETIC_PARAM_CLASSIFICATIONS(object):
    # WARNING(allie): infinite_synthetic should probably be flagged differently for test data.
    data = {'synthetic_generator_n_instances_per_semantic_id', 'portrait', 'infinite_synthetic'}


def get_default_train_config():
    _default_config = generic_cfg.get_default_train_config()
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
        blob_size=None
    )
    return _default_config


def get_default_test_config():
    _default_config = generic_cfg.get_default_test_config()
    _default_config.update(
        dataset='synthetic',
        dataset_path='/tmp/',
        n_images_test=100,
        one_dimension=None,  # {'x', 'y'}
        semantic_subset=None,
        synthetic_generator_n_instances_per_semantic_id=2,
        img_size=None,
        portrait=None,
        blob_size=None
    )
    return _default_config


train_configurations = {
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
        interval_validate=50,
        n_images_train=1,
        n_images_val=1,
        write_instance_metrics=True,
        export_activations=False,
        loss_type='soft_iou',
        size_average=False,
        lr=1.0e-3,
        max_iteration=500,
        sampler='overfit_1'
    ),
    'test_overfit_1_xent': dict(
        interval_validate=50,
        n_images_train=1,
        n_images_val=1,
        write_instance_metrics=True,
        export_activations=False,
        loss_type='cross_entropy',
        size_average=1,
        lr=1.0e-5,
        max_iteration=500,
        sampler='overfit_1'
    ),
    'semantic': dict(
        map_to_semantic=True
        )
}

test_configurations = {
    0: dict(),
}


# TODO(allie): Assert all synthetic params are classified

# def assert_all_cfg_keys_classified():
#     keys = list(_default_train_config.keys())
#     param_groups = [x for x in set(list(SYNTHETIC_PARAM_CLASSIFICATIONS.__dict__.keys()) +
#                                    list(generic_cfg.PARAM_CLASSIFICATIONS.__dict__.keys())) if x[0] != '_']
#     unclassified_keys = []
#     for k in keys:
#         if not any([k in getattr(PARAM_CLASSIFICATIONS, param_group)
#                     for param_group in param_groups]):
#             unclassified_keys.append(k)
#     if len(unclassified_keys) > 0:
#         raise Exception('The following parameters have not yet been classified in PARAM_CLASSIFICATIONS: {}'.format(
#             unclassified_keys))
#
#
# assert_all_cfg_keys_classified()

