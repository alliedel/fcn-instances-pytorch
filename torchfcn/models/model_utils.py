import numpy as np
import torch
import torchfcn


def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        torchfcn.models.FCN32s,
        torchfcn.models.FCN16s,
        torchfcn.models.FCN8s,
        torchfcn.models.FCN8sInstance,
    )
    for m in model.modules():
        # import ipdb; ipdb.set_trace()
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            import ipdb; ipdb.set_trace()
            raise ValueError('Unexpected module: %s' % str(m))


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
