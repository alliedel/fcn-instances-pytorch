import torch
import torch.nn as nn
from torchfcn.models import model_utils
################################################################################
'''
    Helper functions
'''


# TODO(allie): Figure out why this model doesn't work for other input image sizes


####################################
### Mask Encoder (single encoder that takes a depth image and predicts segmentation masks)
# Model that takes in "depth/point cloud" to generate "k"-channel masks
class SimpleSymmetricFCN(nn.Module):
    def __init__(self, n_semantic_classes_with_background=21, n_max_per_class=3,
                 background_classes=(0,),
                 void_classes=(-1,), map_to_semantic=False):

        ##### Copied this from FCN8s_instance.py, not sure how much of it is needed
        if type(background_classes) is tuple:
            background_classes = list(background_classes)
        if type(void_classes) is tuple:
            void_classes = list(void_classes)
        assert len(background_classes) == 1 and background_classes[0] == 0, NotImplementedError
        assert len(void_classes) == 1 and void_classes[0] == -1, NotImplementedError
        super(SimpleSymmetricFCN, self).__init__()

        self.map_to_semantic = map_to_semantic

        self.semantic_instance_class_list = [0]
        for semantic_class in range(1, n_semantic_classes_with_background):
            self.semantic_instance_class_list += [
                semantic_class for _ in range(n_max_per_class)]

        self.n_classes = len(self.semantic_instance_class_list)
        self.n_semantic_classes = n_semantic_classes_with_background
        self.n_max_per_class = n_max_per_class

        self.instance_to_semantic_mapping_matrix = torch.zeros((self.n_classes, self.n_semantic_classes)).float()
        for instance_idx, semantic_idx in enumerate(self.semantic_instance_class_list):
            self.instance_to_semantic_mapping_matrix[instance_idx,
                                                     semantic_idx] = 1

        ###### Choose type of convolution
        ConvType = model_utils.BasicConv2D
        DeconvType = model_utils.BasicDeconv2D
        use_bn, nonlinearity, wide = True, 'prelu', False

        ###### Encoder
        # Create conv-encoder (large net => 5 conv layers with pooling)
        chn = [32, 64, 128, 256, 256, 256] if wide else [16, 16, 32, 64, 128, 128]  # Num channels
        self.conv1 = ConvType(3, chn[0], kernel_size=9, stride=1, padding=4,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 9x9, 140, 250
        self.conv2 = ConvType(chn[0], chn[1], kernel_size=7, stride=1, padding=3,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 7x7, 70, 125
        self.conv3 = ConvType(chn[1], chn[2], kernel_size=5, stride=1, padding=2,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 5x5, 35, 62
        self.conv4 = ConvType(chn[2], chn[3], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 17, 31
        self.conv5 = ConvType(chn[3], chn[4], kernel_size=3, stride=1, padding=1,
                              use_pool=True, use_bn=use_bn, nonlinearity=nonlinearity)  # 3x3, 8, 15

        ###### Mask Decoder
        # Create deconv-decoder (FCN style, has skip-add connections to conv outputs)
        self.conv1x1 = ConvType(chn[4], chn[4], kernel_size=1, stride=1, padding=0,
                                use_pool=False, use_bn=use_bn, nonlinearity=nonlinearity)  # 8, 15
        self.deconv1 = DeconvType(chn[4], chn[3], kernel_size=3, stride=2, padding=0,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 17, 31
        self.deconv2 = DeconvType(chn[3], chn[2], kernel_size=(3,4), stride=2, padding=(0,1),
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 35, 62
        self.deconv3 = DeconvType(chn[2], chn[1], kernel_size=(6,5), stride=2, padding=(2,1),
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 70, 125
        self.deconv4 = DeconvType(chn[1], chn[0], kernel_size=6, stride=2, padding=2,
                                  use_bn=use_bn, nonlinearity=nonlinearity)  # 140, 250
        self.deconv5 = nn.ConvTranspose2d(chn[0], self.n_classes, kernel_size=(7,8), stride=2,
                                          padding=(2,3))  # 281, 500
        if self.map_to_semantic:
            self.conv1x1_instance_to_semantic = nn.Conv2d(in_channels=self.n_classes,
                                                          out_channels=self.n_semantic_classes,
                                                          kernel_size=1, bias=False)

    def forward(self, x):
        # Run conv-encoder to generate embedding
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)

        # Run mask-decoder to predict a smooth mask
        m = self.conv1x1(c5)
        m = self.deconv1(m, c4)
        m = self.deconv2(m, c3)
        m = self.deconv3(m, c2)
        m = self.deconv4(m, c1)
        m = self.deconv5(m)

        if self.map_to_semantic:
            m = self.conv1x1_instance_to_semantic(m)

        # Return masks
        return m
