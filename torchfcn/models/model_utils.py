import torchfcn
from torch import nn


# Choose non-linearities
def get_nonlinearity(nonlinearity):
    if nonlinearity == 'prelu':
        return nn.PReLU()
    elif nonlinearity == 'relu':
        return nn.ReLU(inplace=True)
    elif nonlinearity == 'tanh':
        return nn.Tanh()
    elif nonlinearity == 'sigmoid':
        return nn.Sigmoid()
    elif nonlinearity == 'elu':
        return nn.ELU(inplace=True)
    elif nonlinearity == 'selu':
        return nn.SELU()
    else:
        assert False, "Unknown non-linearity: {}".format(nonlinearity)


### Basic Conv + Pool + BN + Non-linearity structure
class BasicConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_pool=False, use_bn=True, nonlinearity='prelu', **kwargs):
        super(BasicConv2D, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if use_pool else None
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)

    # Convolution -> Pool -> BN -> Non-linearity
    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)


### Basic Deconv + (Optional Skip-Add) + BN + Non-linearity structure
class BasicDeconv2D(nn.Module):
    def __init__(self, in_channels, out_channels, use_bn=True, nonlinearity='prelu', **kwargs):
        super(BasicDeconv2D, self).__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001) if use_bn else None
        self.nonlin = get_nonlinearity(nonlinearity)

    # BN -> Non-linearity -> Deconvolution -> (Optional Skip-Add)
    def forward(self, x, y=None):
        if y is not None:
            x = self.deconv(x) + y  # Skip-Add the extra input
        else:
            x = self.deconv(x)
        if self.bn:
            x = self.bn(x)
        return self.nonlin(x)


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
