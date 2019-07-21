# ---------------------------------------------------------------------------
# Unified Panoptic Segmentation Network
#
# Copyright (c) 2018-2019 Uber Technologies, Inc.
#
# Licensed under the Uber Non-Commercial License (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at the root directory of this project.
#
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Written by Yuwen Xiong
# ---------------------------------------------------------------------------

import torch
import torch.nn as nn
from upsnet.config.config import config
from upsnet.models.fpn import FPN
import upsnet.models.fpn
print(upsnet.models.fpn.__file__)
from upsnet.models.resnet import get_params, resnet_rcnn, ResNetBackbone
import warnings
from torch.nn.parameter import Parameter
import numpy as np
import torch.nn.functional as F


class Resnet(nn.Module):

    def __init__(self, backbone_depth):
        super(Resnet, self).__init__()
        # self.num_classes = config.dataset.num_classes
        # self.num_seg_classes = config.dataset.num_seg_classes
        # self.num_reg_classes = (2 if config.network.cls_agnostic_bbox_reg else config.dataset.num_classes)

        # backbone net
        self.backbone_depth = backbone_depth
        self.resnet_backbone = ResNetBackbone(backbone_depth)
        # self.fpn = FPN(feature_dim=config.network.fpn_feature_dim, with_norm=config.network.fpn_with_norm,
        #                upsample_method=config.network.fpn_upsample_method)

        self.panoptic_loss = nn.CrossEntropyLoss(ignore_index=255, reduce=False)

        self.initialize()

    def initialize(self):
        pass

    def forward(self, data):
        res2, res3, res4, res5 = self.resnet_backbone(data)
        # fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6 = self.fpn(res2, res3, res4, res5)

        # generate gt for panoptic head
        # with torch.no_grad():
        #     if self.enable_void:
        #         panoptic_gt = self.mask_matching(label['seg_gt_4x'], label['mask_gt'], keep_inds=keep_inds)
        #     else:
        #         panoptic_gt = self.mask_matching(label['seg_gt_4x'], label['mask_gt'])

        # # Calc panoptic logits
        # seg_logits, seg_inst_logits = self.seg_term(cls_idx, fcn_output['fcn_score'], gt_rois)
        # mask_logits = self.mask_term(mask_score, gt_rois, cls_idx, fcn_output['fcn_score'])
        #
        # if self.enable_void:
        #     void_logits = torch.max(fcn_output['fcn_score'][:, (config.dataset.num_seg_classes - config.dataset.num_classes + 1):, ...], dim=1, keepdim=True)[0] - torch.max(seg_inst_logits, dim=1, keepdim=True)[0]
        #     inst_logits = seg_inst_logits + mask_logits
        #     panoptic_logits = torch.cat([seg_logits, inst_logits, void_logits], dim=1)
        # else:
        #     panoptic_logits = torch.cat([seg_logits, (seg_inst_logits + mask_logits)], dim=1)

        # Panoptic head loss
        # panoptic_acc = self.calc_panoptic_acc(panoptic_logits, panoptic_gt)
        # panoptic_loss = self.panoptic_loss(panoptic_logits, panoptic_gt)
        # panoptic_loss = panoptic_loss.mean()
        #
        # output = {
        #     'panoptic_loss': panoptic_loss.unsqueeze(0),
        #     'panoptic_accuracy': panoptic_acc.unsqueeze(0),
        # }
        # concat_features = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6])

        concat_features = torch.cat([self.upsample(f, size=data.shape[2:4], upsample_method='nearest')
                                     for f in [res2, res3, res4, res5]], dim=1)
        return concat_features

    def upsample(self, tensor, size, upsample_method):
        return F.interpolate(tensor, size=size, mode=upsample_method,
                             align_corners=False if upsample_method == 'bilinear' else None)

    def calc_panoptic_acc(self, panoptic_logits, gt):
        _, output_cls = torch.max(panoptic_logits.data, 1, keepdim=True)
        ignore = (gt == 255).long().sum()
        correct = (output_cls.view(-1) == gt.data.view(-1)).long().sum()
        total = (gt.view(-1).shape[0]) - ignore
        assert total != 0
        panoptic_acc = correct.float() / total.float()
        return panoptic_acc

    def get_params_lr(self):
        ret = []
        gn_params = []
        gn_params_name = []
        for n, m in self.named_modules():
            if isinstance(m, nn.GroupNorm):
                gn_params.append(m.weight)
                gn_params.append(m.bias)
                gn_params_name.append(n + '.weight')
                gn_params_name.append(n + '.bias')

        ret.append({'params': gn_params, 'lr': 1, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['resnet_backbone.res3', 'resnet_backbone.res4',
                                                            'resnet_backbone.res5'], ['weight'])], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['resnet_backbone.res3', 'resnet_backbone.res4',
                                                            'resnet_backbone.res5'], ['bias'])], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['fpn'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['fpn'], ['bias'], exclude=gn_params_name)], 'lr': 2,
                    'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['rcnn'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['rcnn'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['mask_branch'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['mask_branch'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['rpn'], ['weight'])], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['rpn'], ['bias'])], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['fcn_head'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['fcn_head'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})

        return ret

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
        self.load_backbone_state_dict(state_dict, resume)

    def load_backbone_state_dict(self, state_dict, resume=False):
        own_state = self.state_dict()
        if 'rcnn.cls_score.weight' in state_dict and own_state['rcnn.cls_score.weight'].shape[0] == 9 and state_dict['rcnn.cls_score.weight'].shape[0] == 81:
            cls_map = {  # Imagenet -> COCO
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
            cls_map = {  # ImageNet->cityscapes
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
                print(Warning('unexpected key "{}" in state_dict'.format(name)))
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
            else:
                print(Warning('While copying the parameter named {}, whose dimensions in the models are'
                              ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                    name, own_state[name].size(), param.size())))

        missing = set(own_state.keys()) - set([self.name_mapping(_, resume) for _ in state_dict.keys()])
        if len(missing) > 0:
            non_worrying_suffixes = ['num_batches_tracked']
            for suffix in non_worrying_suffixes:
                num_missing = len(missing)
                missing = [m for m in missing if not m.endswith(suffix)]
                if len(missing) != num_missing:
                    print(Warning('Missing params ending in {}'.format(suffix)))
            if len(missing) > 0:
                print(Warning('missing keys in state_dict: "{}"'.format(missing)))
        copied = set(own_state.keys()).intersection(set([self.name_mapping(_, resume) for _ in state_dict.keys()]))


def resnet_101_upsnet():
    return Resnet([3, 4, 23, 3])


def resnet_50_upsnet():
    return Resnet([3, 4, 6, 3])
