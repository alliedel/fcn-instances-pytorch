class PARAM_CLASSIFICATIONS(object):
    optim = {'optim', 'max_iteration', 'lr', 'momentum', 'weight_decay', 'clip'}
    export = {'interval_validate', 'export_activations', 'activation_layers_to_export', 'write_instance_metrics'}
    loss = {'matching', 'size_average'}
    data = {'semantic_only_labels', 'set_extras_to_void', 'semantic_subset', 'ordering', 'sampler', 'dataset',
            'dataset_instance_cap'}
    problem_config = {'n_instances_per_class', 'single_instance'}
    model = {'initialize_from_semantic', 'bottleneck_channel_capacity', 'score_multiplier', 'freeze_vgg',
             'map_to_semantic', 'augment_semantic', 'use_conv8', 'use_attn_layer'}


# NOTE(allie): Do not directly access this dictionary unless you want to change it for *every* module that imports
# this one.  Ran into issues not copying this dictionary when I started changing it, and it changes all the config
# dictionaries.
_default_config = dict(
    # loss
    matching=True,
    size_average=True,

    # optim
    optim='sgd',
    max_iteration=100000,
    lr=1.0e-12,
    momentum=0.99,
    weight_decay=0.0005,
    clip=1e20,

    # export
    interval_validate=4000,
    export_activations=False,
    activation_layers_to_export=('conv1_1',
                                 'pool3', 'pool4', 'pool5', 'drop6', 'fc7', 'drop7',
                                 'upscore8', 'conv1x1_instance_to_semantic'),
    write_instance_metrics=True,

    # data
    dataset=None,
    dataset_instance_cap='match_model',  #
    semantic_subset=None,
    ordering=None,  # 'lr'
    sampler=None,
    resize=False,
    resize_size=None,
    # semantic_only_labels=False,
    # set_extras_to_void=True,

    # problem_config
    n_instances_per_class=None,
    single_instance=False,  # map_to_single_instance_problem

    # model
    initialize_from_semantic=False,
    bottleneck_channel_capacity=None,
    score_multiplier=None,
    freeze_vgg=False,
    map_to_semantic=False,
    augment_semantic=False,
    use_conv8=False,
    use_attn_layer=False,
)


def get_default_config():
    return _default_config.copy()


def assert_all_cfg_keys_classified():
    keys = list(_default_config.keys())
    param_groups = [x for x in list(PARAM_CLASSIFICATIONS.__dict__.keys()) if x[0] != '_']
    for k in keys:
        assert any([k in getattr(PARAM_CLASSIFICATIONS, param_group)
                    for param_group in param_groups])


assert_all_cfg_keys_classified()
