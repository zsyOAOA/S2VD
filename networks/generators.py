#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-08-10 15:52:29

import torch
from torch import nn
from math import ceil, sqrt

class GeneratorState(nn.Module):
    def __init__(self, latent_size=64, state_size=128, motion_size=64, num_feature=128):
        '''
        Input:
            latent_size: dim of latent variable z
            state_size: dim of state variable s
            num_feature: number of units of the hidden layer
        '''
        super(GeneratorState, self).__init__()
        self.motion_size = motion_size
        self.linear1 = nn.Sequential(
                nn.Linear(latent_size+state_size, num_feature, bias=True),
                nn.ReLU(True)
                )
        if motion_size > 0:
            self.linear2 = nn.Sequential(
                    nn.Linear(motion_size, num_feature, bias=True),
                    nn.ReLU(True)
                    )
        self.linear3 = nn.Sequential(
                nn.Linear(num_feature, state_size, bias=True),
                nn.Tanh()
                )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=sqrt(2))
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, z, state, motion=None):
        x1 = self.linear1(torch.cat([z, state], dim=1))
        if self.motion_size > 0 or motion is not None:
            x2 = self.linear2(motion)
            state_next = self.linear3(x1+x2)
        else:
            state_next = self.linear3(x1)

        return state_next

class GeneratorRain(nn.Module):
    def __init__(self, im_size,
                       out_channels=3,
                       filter_size=3,
                       state_size=128,
                       up_scale=2,
                       num_feature=64):
        '''
        Input:
            im_size: 2-dim tuple or list, [h, w]
            filter_size: integer, filter size default 5
            state_size: dim of state variable s
            up_scale: scale of the last PixelShuffle layer
            num_feature: number of feature maps of the middle convolution layers
        '''
        super(GeneratorRain, self).__init__()
        self.height, self.width = im_size
        self.height_down = ceil(self.height / up_scale)
        self.width_down = ceil(self.width / up_scale)

        self.linear_layer = nn.Sequential(
            nn.Linear(state_size, self.height_down*self.width_down, bias=True),
            nn.ReLU(inplace=True)
                )

        self.body= nn.Sequential(
                nn.Conv2d(in_channels=1,
                          out_channels=num_feature*8,
                          kernel_size=filter_size,
                          stride=1,
                          padding=int((filter_size-1)/2),
                          bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num_feature*8,
                          out_channels=num_feature*4,
                          kernel_size=filter_size,
                          stride=1,
                          padding=int((filter_size-1)/2),
                          bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num_feature*4,
                          out_channels=num_feature*2,
                          kernel_size=filter_size,
                          stride=1,
                          padding=int((filter_size-1)/2),
                          bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=num_feature*2,
                          out_channels=num_feature*(up_scale**2),
                          kernel_size=filter_size,
                          stride=1,
                          padding=int((filter_size-1)/2),
                          bias=True),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(up_scale),
                nn.Conv2d(in_channels=num_feature,
                          out_channels=out_channels,
                          kernel_size=filter_size,
                          stride=1,
                          padding=int((filter_size-1)/2),
                          bias=True),
                nn.ReLU(inplace=True)
                )

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.linear_layer(x)
        x = x.view([-1, 1, self.height_down, self.width_down])
        rain = self.body(x)[:, :, :self.height, :self.width]

        return rain
