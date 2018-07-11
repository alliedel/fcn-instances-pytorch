# cityscapes/config.py

default_configuration = dict(
    max_iteration=10000,
    lr=1.0e-5,
    weight_decay=5e-6,
    interval_validate=1000,
    n_max_per_class=10,
    n_training_imgs=1000,
    n_validation_imgs=50,
    batch_size=1,
    recompute_optimal_loss=False,
    size_average=True,
    val_on_train=True,
    resize_size=[256, 512],
    map_to_semantic=False,
    no_inst=False,  # convert all instance classes to semantic classes
    matching=True,
    semantic_subset=None)

CONFIG_KEY_REPLACEMENTS_FOR_FILENAME = {'max_iteration': 'itr',
                                        'weight_decay': 'decay',
                                        'n_training_imgs': 'n_train',
                                        'n_validation_imgs': 'n_val',
                                        'recompute_optimal_loss': 'recomp',
                                        'size_average': 'size_avg',
                                        'map_to_semantic': 'mts',
                                        'interval_validate': 'int_val',
                                        'resize_size': 'sz',
                                        'n_max_per_class': 'n_per',
                                        'semantic_subset': 'sem_set',
                                        'val_on_train': 'vot',
                                        'matching': 'match'}

configurations = {
    0: dict(),
    1: dict(
        batch_size=10),
    2: dict(
        n_training_imgs=10),
    3: dict(
        n_training_imgs=100),
    4: dict(
        recompute_optimal_loss=True,
        size_average=True
    ),
    5: dict(
        recompute_optimal_loss=True,
        size_average=False
    ),
    6: dict(
        recompute_optimal_loss=False,
        size_average=False
    ),
    7: dict(
        batch_size=2),
    8: dict(
        max_iteration=100000),
    9: dict(
        n_max_per_class=1,
        interval_validate=100,
        no_inst=True,
        matching=False
    ),
    10: dict(
        n_max_per_class=10,
        interval_validate=100
    )
}


def create_config_copy(config_dict, config_key_replacements=CONFIG_KEY_REPLACEMENTS_FOR_FILENAME):
    cfg_print = config_dict.copy()
    for key, replacement_key in config_key_replacements.items():
        if key in cfg_print:
            cfg_print[replacement_key] = cfg_print.pop(key)
    return cfg_print


def get_config(config_idx):
    cfg = default_configuration
    cfg.update(configurations[config_idx])
    return cfg
