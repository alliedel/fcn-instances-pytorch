import os.path as osp

try:
    import fcn
except ImportError:
    fcn = None
import torch
import torch.nn as nn
from torchfcn import instance_utils

from torchfcn.models import model_utils

DEFAULT_SAVED_MODEL_PATH = osp.expanduser('~/data/models/pytorch/fcn8s-instance.pth')


# TODO(allie): print out flops so you know how long things should take

# TODO(allie): Handle case when extra instances (or semantic segmentation) lands in channel 0

DEBUG = True


class FCN8sInstanceNotAtOnce(nn.Module):

    def __init__(self, n_classes=None, semantic_instance_class_list=None, map_to_semantic=False,
                 include_instance_channel0=False, bottleneck_channel_capacity=None, score_multiplier_init=None):
        """
        n_classes: Number of output channels
        map_to_semantic: If True, n_semantic_classes must not be None.
        include_instance_channel0: If True, extras are placed in instance channel 0 for each semantic class (otherwise
        we don't allocate space for a channel like this)
        bottleneck_channel_capacity: n_classes (default); 'semantic': n_semantic_classes', some number
        """
        if include_instance_channel0:
            raise NotImplementedError
        super(FCN8sInstanceNotAtOnce, self).__init__()
        if n_classes is None:
            assert semantic_instance_class_list is not None, 'either n_classes or semantic_instance_class_list must ' \
                                                             'be specified.'
        else:
            self.n_classes = n_classes
        self.map_to_semantic = map_to_semantic
        self.score_multiplier_init = score_multiplier_init
        if semantic_instance_class_list is None:
            assert not map_to_semantic, ValueError('need semantic_instance_class_list to map to semantic')
            # self.semantic_instance_class_list = np.arange(n_classes, dtype=int)
            self.semantic_instance_class_list = list(range(n_classes))
        else:
            self.semantic_instance_class_list = semantic_instance_class_list
            if n_classes is not None:
                assert n_classes == len(semantic_instance_class_list), \
                    'n_classes does not math the length of the semantic_instance_class_list you passed in.  ' \
                    'Either leave it unspecified or check n_classes (number of instance classes): {}'.format(n_classes)
            self.n_classes = len(semantic_instance_class_list)
        self.instance_to_semantic_mapping_matrix = \
            instance_utils.get_instance_to_semantic_mapping_from_sem_inst_class_list(
                self.semantic_instance_class_list, as_numpy=False)
        self.n_semantic_classes = self.instance_to_semantic_mapping_matrix.size(1)

        if bottleneck_channel_capacity is None:
            self.bottleneck_channel_capacity = self.n_classes
        elif isinstance(bottleneck_channel_capacity, str):
            assert bottleneck_channel_capacity == 'semantic', ValueError('Did not recognize '
                                                                         'bottleneck_channel_capacity {}')
            self.bottleneck_channel_capacity = self.n_semantic_classes
        else:
            assert bottleneck_channel_capacity == int(bottleneck_channel_capacity), ValueError(
                'bottleneck_channel_capacity must be an int')
            self.bottleneck_channel_capacity = int(bottleneck_channel_capacity)

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
        self.score_fr = nn.Conv2d(4096, self.bottleneck_channel_capacity, 1)
        # H/32 x W/32 x n_semantic_cls
        self.score_pool3 = nn.Conv2d(256, self.bottleneck_channel_capacity, 1)
        # H/32 x W/32 x n_semantic_cls
        self.score_pool4 = nn.Conv2d(512, self.bottleneck_channel_capacity, 1)

        self.upscore2 = nn.ConvTranspose2d(  # H/16 x W/16 x n_semantic_cls
            self.bottleneck_channel_capacity, self.bottleneck_channel_capacity, kernel_size=4, stride=2, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(  # H x W x n_semantic_cls
            self.bottleneck_channel_capacity, self.bottleneck_channel_capacity, kernel_size=4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(  # H/2 x W/2 x n_semantic_cls
            self.bottleneck_channel_capacity, self.n_classes, kernel_size=16, stride=8, bias=False)
        if self.score_multiplier_init is not None:
            self.score_multiplier1x1 = nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1, bias=True)
        if self.map_to_semantic:
            self.conv1x1_instance_to_semantic = nn.Conv2d(in_channels=self.n_classes,
                                                          out_channels=self.n_semantic_classes,
                                                          kernel_size=1, bias=False)
        self._initialize_weights()

    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
            9:9 + upscore_pool4.size()[2],
            9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)

        if self.map_to_semantic:
            h = self.conv1x1_instance_to_semantic(h)

        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h

    def copy_params_from_fcn8s(self, fcn16s):
        raise NotImplementedError('function not yet adapted for instance rather than semantic networks (gotta copy '
                                  'weights to each instance from the same semantic class)')
        # for name, l1 in fcn16s.named_children():
        #     try:
        #         l2 = getattr(self, name)
        #         l2.weight  # skip ReLU / Dropout
        #     except Exception:
        #         continue
        #     assert l1.weight.size() == l2.weight.size()
        #     l2.weight.data.copy_(l1.weight.data)
        #     if l1.bias is not None:
        #         assert l1.bias.size() == l2.bias.size()
        #         l2.bias.data.copy_(l1.bias.data)

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
                m.weight.data.zero_()
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                if m.in_channels == m.out_channels:
                    initial_weight = model_utils.get_upsampling_weight(
                        m.in_channels, m.out_channels, m.kernel_size[0])
                else:
                    initial_weight = model_utils.get_non_symmetric_upsampling_weight(
                        m.in_channels, m.out_channels, m.kernel_size[0],
                        semantic_instance_class_list=self.semantic_instance_class_list)
                m.weight.data.copy_(initial_weight)
        if self.score_multiplier_init:
            self.score_multiplier1x1.weight.data.zero_()
            for ch in range(self.score_multiplier1x1.weight.size(1)):
                self.score_multiplier1x1.weight.data[ch, ch] = self.score_multiplier_init
            self.score_multiplier1x1.bias.data.zero_()

    def copy_params_from_vgg16(self, vgg16):
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

    def copy_params_from_semantic_equivalent_of_me(self, semantic_model):
        if self.bottleneck_channel_capacity != self.n_semantic_classes:
            conv2d_with_repeated_channels = ['score_fr', 'score_pool3', 'score_pool4']
            conv2dT_with_repeated_channels = ['upscore2', 'upscore8', 'upscore_pool4']
        else:
            conv2d_with_repeated_channels = []
            conv2dT_with_repeated_channels = ['upscore8']
        module_types_to_ignore = [nn.ReLU, nn.MaxPool2d, nn.Dropout2d]
        module_names_to_ignore = ['score_multiplier1x1']
        # check whether this has the right number of channels to be the semantic version of me
        assert self.semantic_instance_class_list is not None, ValueError('I must know which semantic classes each of '
                                                                         'my instance channels map to in order to '
                                                                         'copy weights.')
        n_semantic_classes = self.n_semantic_classes
        last_layer_name = 'upscore8'
        last_features = getattr(semantic_model, last_layer_name)
        if last_features.weight.size(1) != n_semantic_classes:
            raise ValueError('The semantic model I tried to copy from has {} output channels, but I need {} channels '
                             'for each of my semantic classes'.format(last_features.weight.size(1), n_semantic_classes))

        for module_name, my_module in self.named_children():
            if module_name in module_names_to_ignore:
                continue
            module_to_copy = getattr(semantic_model, module_name)
            if module_name in conv2d_with_repeated_channels:
                for p_name, my_p in my_module.named_parameters():
                    p_to_copy = getattr(module_to_copy, p_name)
                    if not all(my_p.size()[c] == p_to_copy.size()[c] for c in range(1, len(my_p.size()))):
                        import ipdb;
                        ipdb.set_trace()
                        raise ValueError('semantic model is formatted incorrectly at layer {}'.format(module_name))
                    if DEBUG:
                        assert my_p.data.size(0) == len(self.semantic_instance_class_list) \
                               and p_to_copy.data.size(0) == n_semantic_classes
                    for inst_cls, sem_cls in enumerate(self.semantic_instance_class_list):
                        # weird formatting because scalar -> scalar not implemented (must be FloatTensor,
                        # so we use slicing)
                        n_instances_this_class = float(sum(
                            [1 if sic == sem_cls else 0 for sic in self.semantic_instance_class_list]))
                        my_p.data[inst_cls:(inst_cls + 1), ...].copy_(p_to_copy.data[sem_cls:(sem_cls + 1),
                                                                      ...] / n_instances_this_class)
            elif module_name in conv2dT_with_repeated_channels:
                assert isinstance(module_to_copy, nn.ConvTranspose2d)
                # assert l1.weight.size() == l2.weight.size()
                # assert l1.bias.size() == l2.bias.size()
                for p_name, my_p in my_module.named_parameters():
                    p_to_copy = getattr(module_to_copy, p_name)
                    if not all(my_p.size()[c] == p_to_copy.size()[c]
                               for c in [0] + list(range(2, len(p_to_copy.size())))):
                        import ipdb; ipdb.set_trace()
                        raise ValueError('semantic model formatted incorrectly for repeating params.')

                    for inst_cls, sem_cls in enumerate(self.semantic_instance_class_list):
                        # weird formatting because scalar -> scalar not implemented (must be FloatTensor,
                        # so we use slicing)
                        my_p.data[:, inst_cls:(inst_cls + 1), ...].copy_(p_to_copy.data[:, sem_cls:(sem_cls + 1), ...])
            elif isinstance(my_module, nn.Conv2d) or isinstance(my_module, nn.ConvTranspose2d):
                assert type(module_to_copy) == type(my_module)
                for p_name, my_p in my_module.named_parameters():
                    p_to_copy = getattr(module_to_copy, p_name)
                    if not my_p.size() == p_to_copy.size():
                        import ipdb; ipdb.set_trace()
                        raise ValueError('semantic model is formatted incorrectly at layer {}'.format(module_name))
                    my_p.data.copy_(p_to_copy.data)
                    assert torch.equal(my_p.data, p_to_copy.data)
            elif any([isinstance(my_module, type) for type in module_types_to_ignore]):
                continue
            else:
                if not module_has_params(my_module):
                    print('Skipping module of type {} (name: {}) because it has no params.  But please place it in '
                          'list of module types to not copy.'.format(type(my_module), my_module))
                    continue
                else:
                    raise Exception('Haven''t handled copying of {}, of type {}'.format(module_name, type(my_module)))

        # Assert that all the weights equal each other
        if DEBUG:
            successfully_copied_modules = []
            unsuccessfully_copied_modules = []
            for module_name, my_module in self.named_children():
                if module_name in module_names_to_ignore:
                    import ipdb; ipdb.set_trace()
                    continue
                module_to_copy = getattr(semantic_model, module_name)
                for i, (my_p, p_to_copy) in enumerate(zip(my_module.named_parameters(), module_to_copy.named_parameters())):
                    assert my_p[0] == p_to_copy[0]
                    if torch.equal(my_p[1].data, p_to_copy[1].data):
                        successfully_copied_modules.append(module_name + ' ' + str(i))
                        continue
                    else:
                        if module_name in (conv2d_with_repeated_channels + conv2dT_with_repeated_channels):
                            are_equal = True
                            for inst_cls, sem_cls in enumerate(self.semantic_instance_class_list):
                                are_equal = torch.equal(my_p[1].data[:, inst_cls, :, :],
                                                        p_to_copy[1].data[:, sem_cls, :, :])
                                if not are_equal:
                                    break
                            if are_equal:
                                successfully_copied_modules.append(module_name + ' ' + str(i))
                            else:
                                unsuccessfully_copied_modules.append(module_name + ' ' + str(i))
            if len(unsuccessfully_copied_modules) > 0:
                raise Exception('modules were not copied correctly: {}'.format(unsuccessfully_copied_modules))
            else:
                print('All modules copied correctly: {}'.format(successfully_copied_modules))
        if self.map_to_semantic:
            self.conv1x1_instance_to_semantic = nn.Conv2d(in_channels=self.n_classes,
                                                          out_channels=self.n_semantic_classes,
                                                          kernel_size=1, bias=False)


def module_has_params(module):
    module_params = module.named_parameters()
    try:
        next(module_params)
        has_params = True
    except StopIteration:
        has_params = False
    return has_params


def FCN8sInstanceNotAtOncePretrained(model_file=DEFAULT_SAVED_MODEL_PATH, **kwargs):
    model = FCN8sInstanceNotAtOnce(**kwargs)
    # state_dict = torch.load(model_file, map_location=lambda storage, location: 'cpu')
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)[
        'model_state_dict']
    model.load_state_dict(state_dict)
    return model


class FCN8sInstanceAtOnce(FCN8sInstanceNotAtOnce):
    pretrained_model = \
        osp.expanduser('~/data/models/pytorch/fcn8s-atonce_from_caffe.pth')

    @classmethod
    def download(cls):
        if fcn is None:
            raise NotImplementedError
        return fcn.data.cached_download(
            url='http://drive.google.com/uc?id=0B9P1L--7Wd2vblE1VUIxV1o2d2M',
            path=cls.pretrained_model,
            md5='bfed4437e941fef58932891217fe6464',
        )

    # @profile
    def forward(self, x):
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4 * 0.01)  # XXX: scaling to train at once
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3 * 0.0001)  # XXX: scaling to train at once
        h = h[:, :,
            9:9 + upscore_pool4.size()[2],
            9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)

        if self.score_multiplier_init:
            h = self.score_multiplier1x1(h)

        h = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()

        return h


def FCN8sInstanceAtOncePretrained(model_file=DEFAULT_SAVED_MODEL_PATH,
                                  n_classes=21, semantic_instance_class_list=None, map_to_semantic=False):
    model = FCN8sInstanceAtOnce(n_classes=n_classes, semantic_instance_class_list=semantic_instance_class_list,
                                map_to_semantic=map_to_semantic)
    # state_dict = torch.load(model_file, map_location=lambda storage, location: 'cpu')
    state_dict = torch.load(model_file, map_location=lambda storage, loc: storage)[
        'model_state_dict']
    model.load_state_dict(state_dict)
    return model
