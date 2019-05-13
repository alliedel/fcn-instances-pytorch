import os.path as osp

from instanceseg.models.instance_common import make_conv_block
from instanceseg.models.model_utils import copy_tensor, copy_conv, module_has_params, get_activations

try:
    import fcn
except ImportError:
    fcn = None

import torch
import torch.nn as nn
from instanceseg.utils import instance_utils

from instanceseg.models import model_utils
from instanceseg.models import resnet

DEFAULT_SAVED_MODEL_PATH = osp.expanduser('~/data/models/pytorch/fcn8s-instance.pth')

# TODO(allie): print out flops so you know how long things should take

# TODO(allie): Handle case when extra instances (or semantic segmentation) lands in channel 0

DEBUG = True

'''
        100 padding for 2 reasons:
            1) support very small input size
            2) allow cropping in order to match size of different layers' feature maps
        Note that the cropped part corresponds to a part of the 100 padding
        Spatial information of different layers' feature maps cannot be align exactly because of cropping, which is bad
'''


class ResNet50Instance(nn.Module):

    def __init__(self, n_instance_classes=None, semantic_instance_class_list=None, map_to_semantic=False,
                 include_instance_channel0=False, bottleneck_channel_capacity=None, score_multiplier_init=None,
                 at_once=True, n_input_channels=3, clip=None, use_conv8=False, use_attention_layer=False):
        """
        n_classes: Number of output channels
        map_to_semantic: If True, n_semantic_classes must not be None.
        include_instance_channel0: If True, extras are placed in instance channel 0 for each semantic class (otherwise
        we don't allocate space for a channel like this)
        bottleneck_channel_capacity: n_classes (default); 'semantic': n_semantic_classes', some number
        """
        super(ResNet50Instance, self).__init__()

        self.backbone = resnet.resnet_50_upsnet()
        self.n_backbone_out_channels = 2048 + 1024 + 512 + 256  # specifically for resnet

        if include_instance_channel0:
            raise NotImplementedError
        if semantic_instance_class_list is None:
            assert n_instance_classes is not None, \
                ValueError('either n_classes or semantic_instance_class_list must be specified.')
            assert not map_to_semantic, ValueError('need semantic_instance_class_list to map to semantic')
        else:
            assert n_instance_classes is None or n_instance_classes == len(semantic_instance_class_list)
            n_instance_classes = len(semantic_instance_class_list)

        if semantic_instance_class_list is None:
            self.semantic_instance_class_list = list(range(n_instance_classes))
        else:
            self.semantic_instance_class_list = semantic_instance_class_list
        self.n_instance_classes = n_instance_classes
        self.map_to_semantic = map_to_semantic
        self.score_multiplier_init = score_multiplier_init
        self.instance_to_semantic_mapping_matrix = \
            instance_utils.get_instance_to_semantic_mapping_from_sem_inst_class_list(
                self.semantic_instance_class_list, as_numpy=False, compose_transposed=True)
        self.n_semantic_classes = self.instance_to_semantic_mapping_matrix.size(0)
        self.n_output_channels = n_instance_classes if not map_to_semantic else self.n_semantic_classes
        self.n_input_channels = n_input_channels
        self.activations = None
        self.activation_layers = []
        self.my_forward_hooks = {}
        self.use_conv8 = use_conv8

        self.conv1x1_to_instance_channels = nn.Conv2d(in_channels=self.n_backbone_out_channels,
                                                      out_channels=self.n_output_channels, kernel_size=1, bias=False)
        self.conv1x1_instance_to_semantic = None if not self.map_to_semantic else \
            nn.Conv2d(in_channels=self.n_instance_classes, out_channels=self.n_output_channels, kernel_size=1,
                      bias=False)

        self._initialize_weights()

    def forward(self, x):
        h = self.backbone(x)
        import ipdb; ipdb.set_trace()
        h = self.conv1x1_to_instance_channels(h)
        if self.map_to_semantic:
            h = self.conv1x1_instance_to_semantic(h)

        return h

    def copy_params_from_fcn8s(self, fcn16s):
        raise NotImplementedError('function not yet adapted for instance rather than semantic networks (gotta copy '
                                  'weights to each instance from the same semantic class)')

    def _initialize_weights(self):
        self.backbone.initialize()

    def store_activation(self, layer, input, output, layer_name):
        if layer_name not in self.activation_layers:
            self.activation_layers.append(layer_name)
        if self.activations is None:
            self.activations = {}
        self.activations[layer_name] = output.data

    def get_activations(self, input, layer_names):
        return get_activations(self, input, layer_names)
