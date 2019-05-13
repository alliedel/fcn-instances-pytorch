import numpy as np
import torch
from graveyard.models import attention_old
from torch.autograd import Variable
from torch.nn import functional as F
from torch import nn

import instanceseg

VGG_CHILDREN_NAMES = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
                      'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
                      'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3',
                      'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
                      'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'pool5',
                      'fc6', 'relu6', 'drop6',
                      'fc7', 'relu7', 'drop7']


def is_nan(val):
    if torch.is_tensor(val):
        return torch.isnan(val)
        # return (val != val).int() + (val == float('inf')).int() + (val == -float('inf')).int()
    elif isinstance(val, np.ndarray):
        return (val != val) + (val == float('inf')) + (val == -float('inf'))
    else:
        return (val != val) or val == float('inf') or val == -float('inf')


def any_nan(tensor):
    nan_elements = is_nan(tensor)
    return nan_elements.any()


def get_clipping_function(min=None, max=None):
    # NOTE(allie): maybe inplace=True?
    return lambda x: F.hardtanh(x, min_val=min, max_val=max)


def freeze_vgg_module_subset(model, vgg_children_names=VGG_CHILDREN_NAMES):
    freeze_children_by_name(model, vgg_children_names)


def freeze_module_list(module_list):
    for my_module in module_list:
        for p_name, my_p in my_module.named_parameters():
            my_p.requires_grad = False


def compare_model_states(model1, model2):
    matching_modules, nonmatching_modules = [], []
    for (module_name, module1), (module_name2, module2) in zip(model1.named_children(), model2.named_children()):
        assert module_name == module_name2, 'Modules from model1 and model2 dont''t matched when named_children() is ' \
                                            'called.'
        module_matches = all([torch.equal(p1, p2) for p1, p2 in zip(module1.parameters(), module2.parameters())])
        if module_matches:
            matching_modules.append(module_name)
        else:
            nonmatching_modules.append(module_name)

    return matching_modules, nonmatching_modules


def freeze_all_children(model):
    """
    Primarily used for debugging, to make sure it doesn't actually learn anything
    """
    for module_name, my_module in model.named_children():
        for p_name, my_p in my_module.named_parameters():
            my_p.requires_grad = False


def freeze_children_by_name(model, module_names_to_freeze, error_for_leftover_modules=True):
    model_children_names = [child[0] for child in model.named_children()]
    if error_for_leftover_modules:
        # Make sure all modules exist
        module_exists = [module_name in model_children_names for module_name in module_names_to_freeze]
        assert all(module_exists), ValueError('Tried to freeze modules that do not exist: {}'.format(
            [module_name for module_name in model_children_names if not module_exists[module_name]]))

    for module_name, my_module in model.named_children():
        if module_name in module_names_to_freeze:
            for p_name, my_p in my_module.named_parameters():
                my_p.requires_grad = False


def get_upsampling_weight(in_channels, out_channels, kernel_size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    assert in_channels == out_channels, ValueError('in_channels must equal out_channels for bilinear initialization')
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    if in_channels == out_channels:
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
    else:
        weight = \
            (filt[np.newaxis, np.newaxis, :, :]).repeat(in_channels, axis=0).repeat(out_channels, axis=1).astype(
                np.float64)
    return torch.from_numpy(weight).float()


def get_non_symmetric_upsampling_weight(in_channels, out_channels, kernel_size, semantic_instance_class_list=None):
    """
    Make a 2D bilinear kernel suitable for upsampling
    semantic_instance_class_list: gives us a list of the classes to expand (from semantic to instance)
    """
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)
    if semantic_instance_class_list is None:
        assert in_channels == out_channels
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype=np.float64)
        weight[range(in_channels), range(out_channels), :, :] = filt
    else:
        if len(semantic_instance_class_list) != out_channels:
            if len(semantic_instance_class_list) == in_channels:
                raise NotImplementedError('I''ve only implemented the expansion of the weights from semantic to '
                                          'instance, not the collapse from instance to semantic.')
            else:
                raise ValueError('I don''t know how to handle initializing bilinear interpolation weights from {} '
                                 'channels to {} channels without semantic_instance_class_list being {} long'.format(
                    in_channels, out_channels, out_channels))
        weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size),
                          dtype=np.float64)
        # NOTE(allie): this for loop could be made more efficient if it becomes a bottleneck.
        for inst_cls_idx, sem_cls in enumerate(semantic_instance_class_list):
            weight[sem_cls, inst_cls_idx, :, :] = filt
        weight = weight
    return torch.from_numpy(weight).float()


def copy_tensor(src, dest):
    """
    Just used to help me remember which direction the copy_ happens
    """
    dest.copy_(src)


def copy_conv(src_conv_module, dest_conv_module):
    assert src_conv_module.weight.size() == dest_conv_module.weight.size()
    assert src_conv_module.bias.size() == dest_conv_module.bias.size()
    copy_tensor(src=src_conv_module.weight.data, dest=dest_conv_module.weight.data)
    copy_tensor(src=src_conv_module.bias.data, dest=dest_conv_module.bias.data)


def module_has_params(module):
    module_params = module.named_parameters()
    try:
        next(module_params)
        has_params = True
    except StopIteration:
        has_params = False
    return has_params


def add_forward_hook(model: nn.Module, layer_name, storage_function=None):
    if storage_function is None:
        try:
            getattr(model, 'store_activation')
        except AttributeError:
            print('Model must have a store_activation function.')
        storage_function = model.store_activation

    try:
        if '.' in layer_name:
            layer_hierarchy = layer_name.split('.')
            suplayer = model
            for i, superlayer_name in enumerate(layer_hierarchy):
                suplayer = getattr(suplayer, superlayer_name)
            layer = suplayer
        else:
            layer = getattr(model, layer_name)
    except AttributeError:
        print('layer names: {}'.format('\n'.join([n for n, _ in model.named_children()])))
        raise AttributeError('Could not find attribute with name {} in {}'.format(layer_name, model.__class__))

    model.my_forward_hooks[layer_name] = layer.register_forward_hook(lambda *args, **kwargs:
                                                                     storage_function(*args, **kwargs,
                                                                                      layer_name=layer_name))


def clear_forward_hooks_and_activations(model):
    model.activations = None
    model.activation_layers = []
    for name, hook in model.my_forward_hooks.items():
        hook.remove()
    model.my_forward_hooks = {}


def get_activations(model, input, layer_names):
    training = model.training
    model.eval()
    for layer_name in layer_names:
        add_forward_hook(model, layer_name)
    model.forward(input)
    activations = model.activations
    clear_forward_hooks_and_activations(model)
    if training:
        model.train()
    return activations


def get_parameters(model: nn.Module, bias=False):
    # TODO(allie): Figure out if we should just not return any requires_grad=False parameters (don't copy batchnorm?)

    for n, p in model.named_parameters():
        if n.endswith('bias') and bias is False:
            continue
        yield p
