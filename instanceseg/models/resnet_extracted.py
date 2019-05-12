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
from upsnet.models.resnet import get_params, resnet_rcnn, ResNetBackbone


class resnet_upsnet(resnet_rcnn):

    def __init__(self, backbone_depth):
        super(resnet_upsnet, self).__init__()
        self.num_classes = config.dataset.num_classes
        self.num_seg_classes = config.dataset.num_seg_classes
        self.num_reg_classes = (2 if config.network.cls_agnostic_bbox_reg else config.dataset.num_classes)

        # backbone net
        self.resnet_backbone = ResNetBackbone(backbone_depth)
        self.fpn = FPN(feature_dim=config.network.fpn_feature_dim, with_norm=config.network.fpn_with_norm,
                       upsample_method=config.network.fpn_upsample_method)

        self.panoptic_loss = nn.CrossEntropyLoss(ignore_index=255, reduce=False)

        self.initialize()

    def initialize(self):
        pass

    def forward(self, data, label=None):

        res2, res3, res4, res5 = self.resnet_backbone(data['data'])
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
        import ipdb; ipdb.set_trace()
        # concat_features = torch.cat([fpn_p2, fpn_p3, fpn_p4, fpn_p5, fpn_p6])
        concat_features = torch.cat([res2, res3, res4, res5])
        return concat_features

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
        ret.append({'params': [_ for _ in get_params(self, ['resnet_backbone.res3', 'resnet_backbone.res4', 'resnet_backbone.res5'], ['weight'])], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['resnet_backbone.res3', 'resnet_backbone.res4', 'resnet_backbone.res5'], ['bias'])], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['fpn'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['fpn'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['rcnn'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['rcnn'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['mask_branch'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['mask_branch'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['rpn'], ['weight'])], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['rpn'], ['bias'])], 'lr': 2, 'weight_decay': 0})
        ret.append({'params': [_ for _ in get_params(self, ['fcn_head'], ['weight'], exclude=gn_params_name)], 'lr': 1})
        ret.append({'params': [_ for _ in get_params(self, ['fcn_head'], ['bias'], exclude=gn_params_name)], 'lr': 2, 'weight_decay': 0})

        return ret


def resnet_101_upsnet():
    return resnet_upsnet([3, 4, 23, 3])


def resnet_50_upsnet():
    return resnet_upsnet([3, 4, 6, 3])
