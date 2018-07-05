# NOTE(allie): Do not directly access this dictionary unless you want to change it for *every* module that imports
# this one.  Ran into issues not copying this dictionary when I started changing it, and it changes all the config
# dictionaries.
_default_config = dict(
    # Loss
    matching=True,
    size_average=True,

    # Optimization
    optim='sgd',
    max_iteration=100000,
    lr=1.0e-12,
    momentum=0.99,
    weight_decay=0.0005,
    clip=1e20,

    # Exports
    interval_validate=4000,
    export_activations=False,
    activation_layers_to_export=('conv1_1',
                                 'pool3', 'pool4', 'pool5', 'drop6', 'fc7', 'drop7',
                                 'upscore8', 'conv1x1_instance_to_semantic'),

    # Dataset
    dataset=None,
    dataset_instance_cap='match_model',
    sampler=None,
    single_instance=False,  # map_to_single_instance_problem
    semantic_only_labels=False,
    semantic_subset=None,

    # Precomputed transformations
    ordering=None,  # 'lr'

    # Network architecture
    add_conv8=False,
    n_instances_per_class=None,
    initialize_from_semantic=False,
    bottleneck_channel_capacity=None,
    score_multiplier=None,
    freeze_vgg=False,
    map_to_semantic=False,
    augment_semantic=False,
)


def get_default_config():
    return _default_config.copy()
