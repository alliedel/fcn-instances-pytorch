import os.path as osp
import os

import torch
import torchvision
import subprocess
from torch import nn
from torch.nn.parameter import Parameter
import numpy as np
import warnings


RESNET_RNN_CACHE_PATH = osp.expanduser('~/data/models/pytorch/resnet-50-caffe.pth')
RESNET_RNN_URL = 'http://www.yuwenxiong.com/pretrained_model/resnet-50-caffe.pth'


def pretrained_resnet_rnn_state_dict(cache_path=None, url=None):
    cache_path = cache_path or RESNET_RNN_CACHE_PATH
    if not osp.exists(cache_path):
        url = url or RESNET_RNN_URL
        child = subprocess.Popen(['curl', url, '-o', cache_path], stdout=subprocess.PIPE)
        stdout = child.communicate()[0]
        exit_code = child.returncode
        assert exit_code == 0
    state_dict = torch.load(cache_path)
    return state_dict


def freeze_backbone(self, freeze_at):
    assert freeze_at > 0
    for p in self.resnet_backbone.conv1.parameters():
        p.requires_grad = False
    self.resnet_backbone.conv1.eval()
    for i in range(2, freeze_at + 1):
        for p in eval('self.resnet_backbone.res{}'.format(i)).parameters():
            p.requires_grad = False
        eval('self.resnet_backbone.res{}'.format(i)).eval()
