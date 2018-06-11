#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: YangMaoke, DuanZhixiang({maokeyang, zhixiangduan}@deepmotion.ai)
# SSD model


import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from collections import OrderedDict

from utils.layers.det.multibox_layer import MultiBoxLayer


DETECTOR_CONFIG = {
    'num_centrals': [256, 128, 128, 128],
    'num_strides': [2, 2, 1, 1],
    'num_padding': [1, 1, 0, 0],
    'vgg_cfg': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512],
}


class SSD(nn.Module):

    def __init__(self, configer):
        super(SSD, self).__init__()

        self.configer = configer
        self.img_size = self.configer.get('data', 'input_size')
        self.num_features = self.configer.get('details', 'num_feature_list')
        self.num_centrals = DETECTOR_CONFIG['num_centrals']
        self.num_paddings = DETECTOR_CONFIG['num_padding']
        self.num_strides = DETECTOR_CONFIG['num_strides']
        self.vgg_features = self.__make_vgg_layers(DETECTOR_CONFIG['vgg_cfg'])
        self.norm4 = L2Norm2d(20)

        max_size = max(self.img_size)

        if max_size < 448:
            self.feature1 = None
            self.feature2 = None
            self.feature3 = None
            self.feature4 = None
            self.feature5 = None
            self.feature_mode = 'small'
        elif 448 < max_size < 896:
            self.feature1 = None
            self.feature2 = None
            self.feature3 = None
            self.feature4 = None
            self.feature5 = None
            self.feature6 = None
            self.feature_mode = 'large'
        else:
            self.feature1 = None
            self.feature2 = None
            self.feature3 = None
            self.feature4 = None
            self.feature5 = None
            self.feature6 = None
            self.feature7 = None
            self.feature_mode = 'huge'

        self.__make_extra_layers(mode=self.feature_mode)
        self.multibox_layer = MultiBoxLayer(configer)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def __make_vgg_layers(cfg, batch_norm=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)

    def __make_extra_layers(self, mode='small'):
        self.feature1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, self.num_features[1], kernel_size=3, padding=6, dilation=6),
            nn.ReLU(),
            nn.Conv2d(self.num_features[1], self.num_features[1], kernel_size=1),
            nn.ReLU(),
        )

        # 'num_features': [512, 1024, 512, 256, 256, 256].
        # 'num_centrals': [256, 128, 128, 128],
        # 'num_strides': [2, 2, 1, 1],
        # 'num_padding': [1, 1, 0, 0],
        self.feature2 = self.__extra_layer(num_in=self.num_features[1], num_out=self.num_features[2],
                                           num_c=self.num_centrals[0], stride=self.num_strides[0],
                                           pad=self.num_padding[0])
        self.feature3 = self.__extra_layer(num_in=self.num_features[2], num_out=self.num_features[3],
                                           num_c=self.num_centrals[1], stride=self.num_strides[1],
                                           pad=self.num_padding[1])
        self.feature4 = self.__extra_layer(num_in=self.num_features[3], num_out=self.num_features[4],
                                           num_c=self.num_centrals[2], stride=self.num_strides[2],
                                           pad=self.num_padding[2])
        self.feature5 = self.__extra_layer(num_in=self.num_features[4], num_out=self.num_features[5],
                                           num_c=self.num_centrals[3], stride=self.num_strides[3],
                                           pad=self.num_padding[3])
        if mode == 'large':
            self.feature6 = self.__extra_layer(num_in=self.num_features[5], num_out=self.num_features[6],
                                               num_c=self.num_centrals[4], stride=self.num_strides[4],
                                               pad=self.num_padding[4])

        elif mode == 'huge':
            self.feature6 = self.__extra_layer(num_in=self.num_features[5], num_out=self.num_features[6],
                                               num_c=self.num_centrals[4], stride=self.num_strides[4],
                                               pad=self.num_padding[4])

            self.feature7 = self.__extra_layer(num_in=self.num_features[6], num_out=self.num_features[7],
                                               num_c=self.num_centrals[5], stride=self.num_strides[5],
                                               pad=self.num_padding[5])

    @staticmethod
    def __extra_layer(num_in, num_out, num_c, stride, pad):
        layer = nn.Sequential(
            nn.Conv2d(num_in, num_c, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(num_c, num_out, kernel_size=3, stride=stride, padding=pad),
            nn.ReLU(),
        )
        return layer

    def forward(self, _input):
        det_feature = []
        feature = self.vgg_features(_input)
        det_feature.append(self.norm4(feature))
        feature = F.max_pool2d(feature, kernel_size=2, stride=2, ceil_mode=True)

        feature = self.feature1(feature)
        det_feature.append(feature)

        feature = self.feature2(feature)
        det_feature.append(feature)

        feature = self.feature3(feature)
        det_feature.append(feature)

        feature = self.feature4(feature)
        det_feature.append(feature)

        feature = self.feature5(feature)
        det_feature.append(feature)

        if self.feature_mode == 'large':
            feature = self.feature6(feature)
            det_feature.append(feature)

        elif self.feature_mode == 'huge':
            feature = self.feature6(feature)
            det_feature.append(feature)

            feature = self.feature7(feature)
            det_feature.append(feature)

        loc_preds, conf_preds = self.multibox_layer(det_feature)

        return loc_preds, conf_preds

    def load_pretrained_weight(self, net):
        blocks = self.vgg_features

        layer_count = 0
        for l1, l2 in zip(net.features, blocks):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                assert l1.weight.size() == l2.weight.size()
                l2.weight.data.copy_(l1.weight.data)
                layer_count += 1

            elif isinstance(l1, nn.BatchNorm2d) and isinstance(l2, nn.BatchNorm2d):
                assert l1.weight.size() == l2.weight.size()
                l2.weight.data.copy_(l1.weight.data)

        print("total %d layers loaded" % layer_count)


class L2Norm2d(nn.Module):
    """L2Norm layer across all channels."""

    def __init__(self, scale):
        super(L2Norm2d, self).__init__()
        self.scale = scale

    def forward(self, x, dim=1):
        """out = scale * x / sqrt(\sum x_i^2)"""

        _sum = x.pow(2).sum(dim).clamp(min=1e-12).rsqrt()
        out = self.scale * x * _sum.unsqueeze(1).expand_as(x)
        return out


def load_pretrained_weight(_model, pth):
    vgg16_bn = models.vgg16_bn()
    vgg16_bn.load_state_dict(torch.load(pth))
    _model.load_pretrained_weight(vgg16_bn)
    return _model


def load_pretrained_model(model, pth, is_local=True):
    if is_local:
        model.load_state_dict(torch.load(pth))
    else:
        weight = torch.load(pth)
        new_state_dict = OrderedDict()
        for k, v in weight.items():
            name = k[7:]  # remove 'module' producing by training on multi gpus
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    return model

if __name__ == "__main__":
    pass
