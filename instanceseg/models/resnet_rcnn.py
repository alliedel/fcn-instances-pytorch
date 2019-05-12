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
RESNET_RNN_URL = 'http://www.yuwenxiong.com/pretrained_model/resnet-50-caffe.pth',


def pretrained_resnet_rnn_state_dict(cache_path=None, url=None):
    cache_path = cache_path or RESNET_RNN_CACHE_PATH
    if osp.exists(cache_path):
        return cache_path
    url = url or RESNET_RNN_URL
    child = subprocess.Popen(['curl', url, '-o', cache_path], stdout=subprocess.PIPE)
    stdout = child.communicate()[0]
    exit_code = child.returncode
    assert exit_code == 0
    state_dict = torch.load(cache_path)
    return state_dict


class resnet_rcnn(nn.Module):

    def name_mapping(self, name, resume=False):
        if resume:
            return name if not name.startswith('module.') else name[len('module.'):]
        if name.startswith('conv1') or name.startswith('bn1'):
            return 'resnet_backbone.conv1.' + name
        return name.replace('layer1', 'resnet_backbone.res2.layers') \
            .replace('layer2', 'resnet_backbone.res3.layers') \
            .replace('layer3', 'resnet_backbone.res4.layers') \
            .replace('layer4', 'resnet_backbone.res5.layers')

    def load_state_dict(self, state_dict, resume=False):
        own_state = self.state_dict()

        if 'rcnn.cls_score.weight' in state_dict and own_state['rcnn.cls_score.weight'].shape[0] == 9 and state_dict['rcnn.cls_score.weight'].shape[0] == 81:
            cls_map = {
                0: 0,  # background
                1: 1,  # person
                2: -1,  # rider, ignore
                3: 3,  # car
                4: 8,  # truck
                5: 6,  # bus
                6: 7,  # train
                7: 4,  # motorcycle
                8: 2,  # bicycle
            }
            for weight_name in ['rcnn.cls_score.weight', 'rcnn.cls_score.bias', 'rcnn.bbox_pred.weight', 'rcnn.bbox_pred.bias', 'mask_branch.mask_score.weight', 'mask_branch.mask_score.bias']:
                mean = state_dict[weight_name].mean().item()
                std = state_dict[weight_name].std().item()
                state_dict[weight_name] = state_dict[weight_name].view(*([81, -1] + list(state_dict[weight_name].shape[1:])))
                weight_blobs = ((np.random.randn(*([9] + list(state_dict[weight_name].shape[1:])))) * std + mean).astype(np.float32)

                for i in range(9):
                    cls = cls_map[i]
                    if cls >= 0:
                        weight_blobs[i] = state_dict[weight_name][cls]
                weight_blobs = weight_blobs.reshape([-1] + list(state_dict[weight_name].shape[2:]))
                state_dict[weight_name] = torch.from_numpy(weight_blobs)

        if 'fcn_head.score.weight' in own_state and 'fcn_head.score.weight' in state_dict and own_state['fcn_head.score.weight'].shape[0] == 19 and state_dict['fcn_head.score.weight'].shape[0] == 133:
            cls_map = {
                0: 20,  # road
                1: 43,  # sidewalk (pavement-merged -> sidewalk)
                2: 49,  # building
                3: 51,  # wall
                4: 37,  # fence
                5: -1,  # pole
                6: 62,  # traffic light
                7: -1,  # traffic sign
                8: 36,  # vegetation (tree-merged -> vegetation)
                9: -1,  # terrain
                10: 39,  # sky
                11: 53,  # person
                12: -1,  # rider
                13: 55,  # car
                14: 60,  # truck
                15: 58,  # bus
                16: 59,  # train
                17: 56,  # motorcycle
                18: 54,  # bicycle
            }
            for weight_name in ['fcn_head.score.weight', 'fcn_head.score.bias']:
                mean = state_dict[weight_name].mean().item()
                std = state_dict[weight_name].std().item()
                weight_blobs = ((np.random.randn(*([19] + list(state_dict[weight_name].shape[1:])))) * std + mean).astype(np.float32)

                for i in range(19):
                    cls = cls_map[i]
                    if cls >= 0:
                        weight_blobs[i] = state_dict[weight_name][cls]
                state_dict[weight_name] = torch.from_numpy(weight_blobs)

        for name, param in state_dict.items():
            name = self.name_mapping(name, resume)
            if name not in own_state:
                warnings.warn('unexpected key "{}" in state_dict'.format(name))
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                warnings.warn('While copying the parameter named {}, whose dimensions in the models are'
                              ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size()))

        missing = set(own_state.keys()) - set([self.name_mapping(_, resume) for _ in state_dict.keys()])
        if len(missing) > 0:
            warnings.warn('missing keys in state_dict: "{}"'.format(missing))

    def get_params_lr(self):
        raise NotImplementedError()

    def freeze_backbone(self, freeze_at):
        assert freeze_at > 0
        for p in self.resnet_backbone.conv1.parameters():
            p.requires_grad = False
        self.resnet_backbone.conv1.eval()
        for i in range(2, freeze_at + 1):
            for p in eval('self.resnet_backbone.res{}'.format(i)).parameters():
                p.requires_grad = False
            eval('self.resnet_backbone.res{}'.format(i)).eval()

