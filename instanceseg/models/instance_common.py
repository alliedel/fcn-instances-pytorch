from collections import OrderedDict

from torch import nn as nn


def isiterable(x):
    try:
        iter(x)
        is_iterable = True
    except TypeError:
        is_iterable = False
    return is_iterable


def get_default_sublayer_names(block_num, n_convs):
    default_names = []
    for c in range(n_convs):
        # default_names.append('convblock{}_conv{}'.format(block_num, c))
        # default_names.append('convblock{}_relu{}'.format(block_num, c))
        default_names.append('conv{}'.format(c))
        default_names.append('relu{}'.format(c))
    # default_names.append('pool{}'.format(block_num))
    default_names.append('pool')
    return default_names


def make_conv_block(in_channels, out_channels, n_convs=3, kernel_sizes: tuple = 3, stride=2,
                    paddings: tuple = 1, nonlinear_type=nn.ReLU, pool_type=nn.MaxPool2d, pool_size=2,
                    layer_names: list or bool=None, block_num=None):
    if layer_names is None:
        layer_names = True
    if layer_names is True:
        assert block_num is not None, 'I need the block number to create a default sublayer name'
        layer_names = get_default_sublayer_names(block_num, n_convs)

    paddings_list = paddings if isiterable(paddings) else [paddings for _ in range(n_convs)]
    kernel_sizes_list = kernel_sizes if isiterable(kernel_sizes) else [kernel_sizes for _ in range(n_convs)]
    in_c = in_channels
    layers = []
    for c in range(n_convs):
        layers.append(nn.Conv2d(in_c, out_channels, kernel_size=kernel_sizes_list[c], padding=paddings_list[c]))
        layers.append(nonlinear_type(inplace=True))
        in_c = out_channels

    layers.append(pool_type(kernel_size=pool_size, stride=stride, ceil_mode=True))
    if layer_names is False:
        return nn.Sequential(*layers)
    else:
        assert len(layer_names) == len(layers)
        layers_with_names = [(name, layer) for name, layer in zip(layer_names, layers)]
        ordered_layers_with_names = OrderedDict(layers_with_names)
        return nn.Sequential(ordered_layers_with_names)