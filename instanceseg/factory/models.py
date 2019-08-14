import os

import torch
from torch import nn

import instanceseg.models
from instanceseg.utils import instance_utils
from instanceseg.models import model_utils
from instanceseg.utils.instance_utils import InstanceProblemConfig
from instanceseg.models import resnet_rcnn

MODEL_OPTIONS = {
    'fcn8': instanceseg.models.FCN8sInstance,
    'resnet50': instanceseg.models.ResNet50Instance
}


def get_model(cfg, problem_config: InstanceProblemConfig, checkpoint_file, semantic_init, cuda):
    n_input_channels = 3 if not cfg['augment_semantic'] else 3 + problem_config.n_semantic_classes
    if cfg['backbone'] == 'fcn8':
        model = MODEL_OPTIONS['fcn8'](
            semantic_instance_class_list=problem_config.model_semantic_instance_class_list,
            map_to_semantic=problem_config.map_to_semantic, include_instance_channel0=False,
            bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'],
            score_multiplier_init=cfg['score_multiplier'],
            n_input_channels=n_input_channels, clip=cfg['clip'], use_conv8=cfg['use_conv8'],
            use_attention_layer=cfg[
                'use_attn_layer'])
    elif cfg['backbone'] == 'resnet50':
        model = instanceseg.models.ResNet50Instance(
            semantic_instance_class_list=problem_config.model_semantic_instance_class_list,
            map_to_semantic=problem_config.map_to_semantic, include_instance_channel0=False,
            bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'],
            score_multiplier_init=cfg['score_multiplier'], n_input_channels=n_input_channels)
    else:
        raise ValueError('Unknown backbone architecture {}.  '.format(cfg['backbone'])
                         + '\nOptions:\n{}'.format(MODEL_OPTIONS.keys()))

    if checkpoint_file is not None:
        checkpoint = torch.load(checkpoint_file)
        state_dict = checkpoint['model_state_dict']
        if list(checkpoint['model_state_dict'].keys())[0].startswith('module') and not hasattr(model, 'module'):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k.replace('module.', '')  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(state_dict)
        start_epoch = checkpoint['epoch']
        start_iteration = checkpoint['iteration']
    else:
        start_epoch, start_iteration = 0, 0
        if cfg['backbone'] == 'fcn8':
            if cfg['initialize_from_semantic']:
                semantic_init_path = os.path.expanduser(semantic_init)
                if not os.path.exists(semantic_init_path):
                    raise ValueError(
                        'I could not find the path {}.  Did you set the path using the '
                        'semantic-init flag?'.format(semantic_init_path))
                semantic_model = instanceseg.models.FCN8sInstance(
                    semantic_instance_class_list=[1 for _ in
                                                  range(problem_config.n_semantic_classes)],
                    map_to_semantic=False, include_instance_channel0=False)
                print('Copying params from preinitialized semantic model')
                checkpoint_file = torch.load(semantic_init_path)
                semantic_model.load_state_dict(checkpoint_file['model_state_dict'])
                model.copy_params_from_semantic_equivalent_of_me(semantic_model)
            else:
                print('Copying params from vgg16')
                vgg16 = instanceseg.models.VGG16(pretrained=True)
                model.copy_params_from_vgg16(vgg16)
        elif cfg['backbone'] == 'resnet50':
            print('Copying params from rcnn resnet')
            model.backbone.load_rnn_resnet_state_dict(resnet_rcnn.pretrained_resnet_rnn_state_dict())

    n_gpus = torch.cuda.device_count()
    print('Using {} GPUS for model: {}'.format(
        n_gpus, [torch.cuda.get_device_name(i) for i in range(n_gpus)]))
    if n_gpus > 1:
        model = nn.DataParallel(model)

    if cuda:
        model = model.cuda()

    if cfg['freeze_vgg']:
        model_utils.freeze_vgg_module_subset(model)
    return model, start_epoch, start_iteration


def get_problem_config_from_labels_table(labels_table, n_instances_by_semantic_id, map_to_semantic=False):
    problem_config = instance_utils.InstanceProblemConfig(labels_table=labels_table,
                                                          n_instances_by_semantic_id=n_instances_by_semantic_id,
                                                          map_to_semantic=map_to_semantic)
    return problem_config
