import os

import torch

import torchfcn
from torchfcn.models import model_utils


def get_model(cfg, problem_config, checkpoint_file, semantic_init, cuda):
    n_input_channels = 3 if not cfg['augment_semantic'] else 3 + problem_config.n_semantic_classes
    model = torchfcn.models.FCN8sInstance(
        semantic_instance_class_list=problem_config.model_semantic_instance_class_list,
        map_to_semantic=problem_config.map_to_semantic, include_instance_channel0=False,
        bottleneck_channel_capacity=cfg['bottleneck_channel_capacity'], score_multiplier_init=cfg['score_multiplier'],
        n_input_channels=n_input_channels, clip=cfg['clip'], add_conv8=cfg['add_conv8'])
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
            semantic_model = torchfcn.models.FCN8sInstance(
                semantic_instance_class_list=[1 for _ in range(problem_config.n_semantic_classes)],
                map_to_semantic=False, include_instance_channel0=False)
            print('Copying params from preinitialized semantic model')
            checkpoint_file = torch.load(semantic_init_path)
            semantic_model.load_state_dict(checkpoint_file['model_state_dict'])
            model.copy_params_from_semantic_equivalent_of_me(semantic_model)
        else:
            print('Copying params from vgg16')
            vgg16 = torchfcn.models.VGG16(pretrained=True)
            model.copy_params_from_vgg16(vgg16)
    if cuda:
        model = model.cuda()

    if cfg['freeze_vgg']:
        model_utils.freeze_vgg_module_subset(model)
    return model, start_epoch, start_iteration