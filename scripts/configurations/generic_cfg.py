default_config = dict(
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
)

