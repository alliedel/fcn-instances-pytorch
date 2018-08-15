import os

import torch

import instanceseg
from instanceseg import instance_utils
from instanceseg.models import model_utils


def get_model(cfg, problem_config, checkpoint_file, semantic_init, cuda):
    n_input_channels = 3 if not cfg['augment_semantic'] else 3 + problem_config.n_semantic_classes
    try:
        model = instanceseg.models.FCN8sInstance(
            semantic_instance_class_list=problem_config.model_semantic_instance_class_list,
            map_to_semantic=problem_config.map_to_semantic, include_instance_channel0=False,
            bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'], score_multiplier_init=cfg['score_multiplier'],
            n_input_channels=n_input_channels, clip=cfg['clip'], use_conv8=cfg['use_conv8'], use_attention_layer=cfg[
                'use_attn_layer'])
    except:
        print('Warning: deprecated.')
        model = instanceseg.models.FCN8sInstance(
            semantic_instance_class_list=problem_config.model_semantic_instance_class_list,
            map_to_semantic=problem_config.map_to_semantic, include_instance_channel0=False,
            bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'], score_multiplier_init=cfg['score_multiplier'],
            n_input_channels=n_input_channels, clip=cfg['clip'])
    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        start_epoch, start_iteration = 0, 0
        if cfg['initialize_from_semantic']:
            semantic_init_path = os.path.expanduser(semantic_init)
            if not os.path.exists(semantic_init_path):
                raise ValueError('I could not find the path {}.  Did you set the path using the semantic-init '
                                 'flag?'.format(semantic_init_path))
            semantic_model = instanceseg.models.FCN8sInstance(
                semantic_instance_class_list=[1 for _ in range(problem_config.n_semantic_classes)],
                map_to_semantic=False, include_instance_channel0=False)
            print('Copying params from preinitialized semantic model')
            checkpoint_file = torch.load(semantic_init_path)
            semantic_model.load_state_dict(checkpoint_file['model_state_dict'])
            model.copy_params_from_semantic_equivalent_of_me(semantic_model)
        else:
            print('Copying params from vgg16')
            vgg16 = instanceseg.models.VGG16(pretrained=True)
            model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    if cfg['freeze_vgg']:
        model_utils.freeze_vgg_module_subset(model)
    return model, start_epoch, start_iteration


def get_problem_config(class_names, n_instances_per_class: int, map_to_semantic=False):
    # 0. Problem setup (instance segmentation definition)
    class_names = class_names
    n_semantic_classes = len(class_names)
    n_instances_by_semantic_id = [1] + [n_instances_per_class for _ in range(1, n_semantic_classes)]
    problem_config = instance_utils.InstanceProblemConfig(n_instances_by_semantic_id=n_instances_by_semantic_id,
                                                          map_to_semantic=map_to_semantic)
    problem_config.set_class_names(class_names)
    return problem_config
