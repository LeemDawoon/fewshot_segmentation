"""
Encoder for few shot segmentation (VGG16)
"""

import torch
import torch.nn as nn
import torchvision


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )
# def double_conv(in_channels, out_channels, kernel, padding):
#     return nn.Sequential(
#         nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
#         nn.ReLU(inplace=True),
#         nn.Conv2d(out_channels, out_channels, kernel, padding=padding),
#         nn.ReLU(inplace=True)
#     )   
class Encoder(nn.Module):
    """
    Encoder for few shot segmentation

    Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
    """
    def __init__(self, in_channels=3, pretrained_path=None):
        super().__init__()
        self.pretrained_path = pretrained_path

        # self.features = nn.Sequential(
        #     self._make_layer(2, in_channels, 64),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     self._make_layer(2, 64, 128),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     self._make_layer(3, 128, 256),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        #     self._make_layer(3, 256, 512),
        #     nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
        #     self._make_layer(3, 512, 512, dilation=2, lastRelu=False),
        # )

        # _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True)

        # self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.down1 = self._make_layer(2, in_channels, 64)
        self.down2 = self._make_layer(2, 64, 128)
        self.down3 = self._make_layer(3, 128, 256)
        self.down4 =self._make_layer(3, 256, 512)
        self.down5 =self._make_layer(3, 512, 512, dilation=2, lastRelu=False)

        self.down2_1x1 = convrelu(128, 128, 1, 0)
        self.down3_1x1 = convrelu(256, 256, 1, 0)
        self.down4_1x1 = convrelu(512, 512, 1, 0)
        self.down5_1x1 = convrelu(512, 512, 1, 0)
        self.last_conv = convrelu(64+128+256+512+512, 512, 1, 0)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        # self.conv_original_size0 = convrelu(3, 64, 3, 1)
        # self.conv_original_size1 = convrelu(64, 64, 3, 1)
        # self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        # self.conv_last = nn.Conv2d(64, n_class, 1)

        self._init_weights()

    def forward(self, x):
        # x.shape torch.Size([6, 3, 224, 224])
        # return self.features(x)
        origin = self.down1(x)
        x = self.pool(origin)

        conv2 = self.down2(x)
        x = self.pool(conv2)
        
        conv3 = self.down3(x)
        x = self.pool(conv3)

        conv4 = self.down4(x)
        x = self.pool(conv4)

        conv5 = self.down5(x)
        x = self.pool(conv5)

        conv2 = self.down2_1x1(conv2)
        conv3 = self.down3_1x1(conv3)
        conv4 = self.down4_1x1(conv4)
        conv5 = self.down5_1x1(conv5)
        # print('origin.shape', origin.shape)
        # print('conv2.shape', conv2.shape)
        # print('conv3.shape', conv3.shape)
        # print('conv4.shape', conv4.shape)
        # print('conv5.shape', conv5.shape)
        # origin.shape torch.Size([6, 64, 224, 224])
        # conv2.shape torch.Size([6, 128, 112, 112])
        # conv3.shape torch.Size([6, 256, 56, 56])
        # conv4.shape torch.Size([6, 512, 28, 28])
        # conv5.shape torch.Size([6, 512, 14, 14])

        conv2 = self.upsample2(conv2)
        conv3 = self.upsample4(conv3)
        conv4 = self.upsample8(conv4)
        conv5 = self.upsample16(conv5)
        # print('origin.shape', origin.shape)
        # print('conv2.shape', conv2.shape)
        # print('conv3.shape', conv3.shape)
        # print('conv4.shape', conv4.shape)
        # print('conv5.shape', conv5.shape)
        # origin.shape torch.Size([6, 64, 224, 224])
        # conv2.shape torch.Size([6, 128, 224, 224])
        # conv3.shape torch.Size([6, 256, 224, 224])
        # conv4.shape torch.Size([6, 512, 224, 224])
        # conv5.shape torch.Size([6, 512, 224, 224])

        x = torch.cat([origin, conv2, conv3, conv4, conv5], dim=1)
        x = self.last_conv(x)
        # x = self.up4(x)
        # x = self.upsample(x)        
        # x = torch.cat([x, conv3], dim=1)    

        # x = self.up3(x)
        # x = self.upsample(x)        
        # x = torch.cat([x, conv2], dim=1)    

        # x = self.up2(x)
        # x = self.upsample(x)        
        # x = torch.cat([x, conv1], dim=1)    

        # x = self.up1(x)

        return x



    def _make_layer(self, n_convs, in_channels, out_channels, dilation=1, lastRelu=True):
        """
        Make a (conv, relu) layer

        Args:
            n_convs:
                number of convolution layers
            in_channels:
                input channels
            out_channels:
                output channels
        """
        layer = []
        for i in range(n_convs):
            layer.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, dilation=dilation, padding=dilation))
            if i != n_convs - 1 or lastRelu:
                layer.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        return nn.Sequential(*layer)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        if self.pretrained_path is not None:
            dic = torch.load(self.pretrained_path, map_location='cpu')
            keys = list(dic.keys())
            new_dic = self.state_dict()
            new_keys = list(new_dic.keys())

            for i in range(26):
                new_dic[new_keys[i]] = dic[keys[i]]

            self.load_state_dict(new_dic)
