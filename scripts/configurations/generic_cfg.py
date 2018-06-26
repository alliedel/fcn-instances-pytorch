# NOTE(allie): Do not directly access this dictionary unless you want to change it for *every* module that imports
# this one.  Ran into issues not copying this dictionary when I started changing it, and it changes all the config
# dictionaries.
_default_config = dict(
    max_iteration=100000,
    lr=1.0e-12,
    momentum=0.99,
    weight_decay=0.0005,
    interval_validate=4000,
    matching=True,
    semantic_only_labels=False,
    n_instances_per_class=None,
    set_extras_to_void=True,
    semantic_subset=None,
    optim='sgd',
    single_instance=False,  # map_to_single_instance_problem
    initialize_from_semantic=False,
    bottleneck_channel_capacity=None,
    size_average=True,
    score_multiplier=None,
    freeze_vgg=False,
    map_to_semantic=False,
    augment_semantic=False,
    clip=1e20,
    export_activations=False,
    activation_layers_to_export=('conv1_1',
                                 'pool3', 'pool4', 'pool5', 'drop6', 'fc7', 'drop7',
                                 'upscore8', 'conv1x1_instance_to_semantic'),
    ordering=None,  # 'lr'
    add_conv8=False,
    sampler=None,
    dataset=None,
    dataset_instance_cap='match_model',  #
)


def get_default_config():
    return _default_config.copy()
