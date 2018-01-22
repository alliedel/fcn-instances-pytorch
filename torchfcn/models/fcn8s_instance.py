import os.path as osp

import fcn
import torch.nn as nn
import torch

from .fcn32s import get_upsampling_weight

from .fcn8s import FCN8s

from numpy import mod

import local_pyutils

logger = local_pyutils.get_logger()


class FCN8sInstance(nn.Module):

    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s-instance.pth')

    def __init__(self, n_semantic_classes_with_background=21, n_max_per_class=3,
                 background_classes=[0],
                 void_classes=[-1], map_to_semantic=False):
        assert len(
            background_classes) == 1 and background_classes[0] == 0, NotImplementedError
        assert len(
            void_classes) == 1 and void_classes[0] == -1, NotImplementedError
        super(FCN8sInstance, self).__init__()

        self.map_to_semantic = map_to_semantic

        self.semantic_instance_class_list = [0]
        for semantic_class in range(n_semantic_classes_with_background - 1):
            self.semantic_instance_class_list += [
                semantic_class for _ in range(n_max_per_class)]

        self.n_classes = len(self.semantic_instance_class_list)
        self.n_semantic_classes = n_semantic_classes_with_background
        self.n_max_per_class = n_max_per_class

        self.instance_to_semantic_mapping_matrix = torch.zeros((self.n_classes,
                                                                self.n_semantic_classes)).float()

        for instance_idx, semantic_idx in enumerate(self.semantic_instance_class_list):
            self.instance_to_semantic_mapping_matrix[instance_idx,
                                                     semantic_idx] = 1

        # conv1
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        try:
            self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        except:
            import ipdb
            ipdb.set_trace()
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(
            kernel_size=2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(
            kernel_size=2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, kernel_size=1)  # H/32 x W/32 x 4096
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # H/32 x W/32 x n_semantic_cls
        self.score_fr = nn.Conv2d(4096, self.n_classes, 1)
        # H/32 x W/32 x n_semantic_cls
        self.score_pool3 = nn.Conv2d(256, self.n_classes, 1)
        # H/32 x W/32 x n_semantic_cls
        self.score_pool4 = nn.Conv2d(512, self.n_classes, 1)

        self.upscore2 = nn.ConvTranspose2d(   # H/16 x W/16 x n_semantic_cls
            self.n_classes, self.n_classes, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(   # H/2 x W/2 x n_semantic_cls
            self.n_classes, self.n_classes, kernel_size=16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(  # H x W x n_semantic_cls
            self.n_classes, self.n_classes, kernel_size=4, stride=2, bias=False)
        if self.map_to_semantic:
            self.conv1x1_instance_to_semantic = nn.Conv2d(in_channels=self.n_classes,
                                                          out_channels=self.n_semantic_classes,
                                                          kernel_size=1, bias=False)
        self._initialize_weights()

    @classmethod
    def download(cls):
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
            path=cls.pretrained_model,
            md5='bfed4437e941fef58932891217fe6464',
        )

    # @profile
    def forward(self, x):
        h = x
        input_size = x.size()
        h, pool3, pool4 = self.reduce_to_fc7(h)

        h = self.upscore(h, pool3, pool4)

        if self.map_to_semantic:
            h = self.conv1x1_instance_to_semantic(h)

        h = self.sample_contiguous_center(h, input_size)

        return h

    def conv1(self, h):
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)
        return h

    def conv2(self, h):
        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)
        return h

    def conv3(self, h):
        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        return h

    def conv4(self, h):
        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        return h

    def conv5(self, h):
        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)
        return h

    def reduce_to_fc7(self, h):
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        pool3 = h  # 1/8
        h = self.conv4(h)
        pool4 = h  # 1/16
        h = self.conv5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        return h, pool3, pool4

    def upscore(self, h, pool3, pool4):
        # Transpose Convolution here ('deconvolution')
        h = self.score_fr(h)
        h = self.upscore2(h)  # ConvTranspose2d, stride=2
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)  # ConvTranspose2d, stride=2
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)  # ConvTranspose2d, stride=8
        return h

    def sample_contiguous_center(self, h, input_size):
        h = h[:, :, 31:31 + input_size[2], 31:31 + input_size[3]].contiguous()
        return h

    def copy_params_from_vgg16(self, vgg16):
        logger.info('Copying params from vgg16')
        features = [
            self.conv1_1, self.relu1_1,
            self.conv1_2, self.relu1_2,
            self.pool1,
            self.conv2_1, self.relu2_1,
            self.conv2_2, self.relu2_2,
            self.pool2,
            self.conv3_1, self.relu3_1,
            self.conv3_2, self.relu3_2,
            self.conv3_3, self.relu3_3,
            self.pool3,
            self.conv4_1, self.relu4_1,
            self.conv4_2, self.relu4_2,
            self.conv4_3, self.relu4_3,
            self.pool4,
            self.conv5_1, self.relu5_1,
            self.conv5_2, self.relu5_2,
            self.conv5_3, self.relu5_3,
            self.pool5,
        ]
        for l1, l2 in zip(vgg16.features, features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data.copy_(l1.weight.data)
                l2.bias.data.copy_(l1.bias.data)
        for i, name in zip([0, 3], ['fc6', 'fc7']):
            l1 = vgg16.classifier[i]
            l2 = getattr(self, name)
            l2.weight.data.copy_(l1.weight.data.view(l2.weight.size()))
            l2.bias.data.copy_(l1.bias.data.view(l2.bias.size()))

    def copy_params_from_fcn8s(self, fcn8s_semantic):
        print(Warning('I haven\'t thoroughly tested this function.'))
        number_semantic_classes_pretrained = [c for c in fcn8s_semantic.named_children()][-1][
            1].weight.size(1)
        if number_semantic_classes_pretrained == self.n_semantic_classes:
            print('Number of semantic classes matches!')
        else:
            raise NotImplementedError('Number of semantic classes does not match.  I don\'t know '
                                      'which ones to copy.')
        logger.info('Copying params from semantic fcn8s')
        for name, l1 in fcn8s_semantic.named_children():
            try:
                l2 = getattr(self, name)
                l2.weight  # skip ReLU / Dropout
            except Exception:
                continue
            if l1.weight.size() == l2.weight.size():
                l2.weight.data.copy_(l1.weight.data)
                if l1.bias is not None:
                    assert l1.bias.size() == l2.bias.size()
                    l2.bias.data.copy_(l1.bias.data)
            else:
                num_repeats_per_dim = [1 if l1.weight.size(d) == l2.weight.size(d)
                                       else self.n_max_per_class
                                       for d in range(len(l1.weight.size()))]
                assert all([repeats == 1 or repeats == self.n_max_per_class for repeats in
                            num_repeats_per_dim])
                try:
                    new_weights_with_background_repeated = \
                        l1.weight.data.repeat(*num_repeats_per_dim)
                    l2.weight.data.copy_(new_weights_with_background_repeated)
                except:
                    import ipdb
                    ipdb.set_trace()
                    raise

    def _initialize_weights(self):
        num_modules = len(list(self.modules()))
        for idx, m in enumerate(self.modules()):
            if idx == num_modules - 1 and self.map_to_semantic:
                assert m == self.conv1x1_instance_to_semantic
                self.conv1x1_instance_to_semantic.weight.data.copy_(
                    self.instance_to_semantic_mapping_matrix.transpose(1, 0))
                self.conv1x1_instance_to_semantic.weight.requires_grad = False  # Fix weights
                print('conv1x1 initialized to have weights of shape {}'.format(
                    self.conv1x1_instance_to_semantic.weight.data.shape))
            elif isinstance(m, nn.Conv2d):
                # m.weight.data.zero_()
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                initial_weight = get_upsampling_weight(
                    m.in_channels, m.out_channels, m.kernel_size[0])
                m.weight.data.copy_(initial_weight)


def FCN8sPretrained(model_file='/home/adelgior/data/models/pytorch/fcn8s_trained_semantic.pth',
                    n_class=21):
    model = FCN8s(n_class=n_class)
    # state_dict = torch.load(model_file, map_location=lambda storage, location: 'cpu')
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)[
        'model_state_dict']
    model.load_state_dict(state_dict)
    return model
