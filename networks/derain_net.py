#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-09-23 15:45:37

import torch.nn as nn
from .sub_blocks import PixelShuffle3D, PixelUnShuffle3D, MyConv3d

class DerainNet(nn.Module):
    def __init__(self, n_channels=3, upscale_factor=2, n_features=32, n_resblocks=8):
        super(DerainNet, self).__init__()

        self.unshuffle = PixelUnShuffle3D(upscale_factor)

        self.head = nn.Sequential(
                    MyConv3d(in_channels=n_channels*(upscale_factor**2),
                             out_channels=n_features,
                             kernel_size=3,
                             stride=1,
                             bias=True),
                    nn.ReLU(True),
                    MyConv3d(in_channels=n_features,
                             out_channels=n_features,
                             kernel_size=3,
                             stride=1,
                             bias=True),
                     )

        # layers resblocks
        self.body = nn.ModuleList()
        for ii in range(n_resblocks):
            self.body.append(ResBlock(n_features, n_features))
            if (ii+1) == n_resblocks:
                self.body.append(nn.BatchNorm3d(n_features))
                self.body.append(nn.ReLU(inplace=True))
                self.body.append(MyConv3d(in_channels=n_features,
                            out_channels=n_channels*(upscale_factor**2),
                            kernel_size=3,
                            stride=1,
                            bias=True))
                self.body.append(PixelShuffle3D(upscale_factor))

        self.tail = nn.Sequential(
                    MyConv3d(in_channels=n_channels,
                             out_channels=n_features,
                             kernel_size=3,
                             stride=1,
                             bias=False),
                    nn.ReLU(True),
                    MyConv3d(in_channels=n_features,
                             out_channels=n_features,
                             kernel_size=3,
                             stride=1,
                             bias=False),
                    nn.ReLU(True),
                    MyConv3d(in_channels=n_features,
                             out_channels=n_channels,
                             kernel_size=3,
                             stride=1,
                             bias=False)
                )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.orthogonal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, images):
        x = self.unshuffle(images)
        x = self.head(x)
        for op in self.body:
            x = op(x)
        out = self.tail(images-x)
        return out

class ResBlock(nn.Module):
    '''
    Res Block: x + conv(ReLU(BN(conv(ReLU(BN(x))))))
    '''
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.res = nn.Sequential(
                nn.BatchNorm3d(in_channels),
                nn.ReLU(True),
                MyConv3d(in_channels, out_channels,
                         kernel_size=3,
                         stride=1,
                         bias=True),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(True),
                MyConv3d(out_channels, out_channels,
                         kernel_size=3,
                         stride=1,
                         bias=True)
                )

    def forward(self, x):
        out = x + self.res(x)
        return out

