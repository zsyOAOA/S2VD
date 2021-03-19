#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-09-23 14:24:24

import torch.nn as nn
import torch.nn.functional as F

def pixel_unshuffle3D(x, upscale_factor):
    '''
    Rearranges elements in a tensor of shape: (B, C, D, rH, rW) to a tensor of shape (B, r^2C, D, H, W).
    :param x: torch tensor
    :param upscale_factor: integer
    '''
    batch_size, channels, frames, in_H, in_W = x.shape

    out_H = in_H // upscale_factor
    out_W = in_W // upscale_factor

    x_temp = x.reshape([batch_size, channels, frames, out_H, upscale_factor, out_W, upscale_factor])

    channels *= (upscale_factor ** 2)
    out_temp = x_temp.permute(0, 1, 4, 6, 2, 3, 5).contiguous()
    out = out_temp.view(batch_size, channels, frames, out_H, out_W)
    return out

class PixelUnShuffle3D(nn.Module):
    '''
    Rearranges elements in a tensor of shape: (B, C, D, rH, rW) to a tensor of shape (B, r^2C, D, H, W).
    '''
    def __init__(self, upscale_factor):
        super(PixelUnShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixel_unshuffle3D(x, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

def pixel_shuffle3D(x, upscale_factor):
    '''
    Rearranges elements in a tensor of shape: (B, r^2C, D, H, W) to a tensor of shape (B, C, D, rH, rW).
    :param x: torch tensor
    :param upscale_factor: integer
    '''
    batch_size, channels, frames, in_H, in_W = x.shape

    channels //= (upscale_factor**2)

    x_temp = x.reshape([batch_size, channels, upscale_factor, upscale_factor, frames, in_H, in_W])
    out_temp = x_temp.permute(0, 1, 4, 5, 2, 6, 3).contiguous()

    out_H = in_H * upscale_factor
    out_W = in_W * upscale_factor
    out = out_temp.view(batch_size, channels, frames, out_H, out_W)
    return out

class PixelShuffle3D(nn.Module):
    '''
    Rearranges elements in a tensor of shape: (B, r^2C, D, H, W) to a tensor of shape (B, C, D, rH, rW).
    '''
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        return pixel_shuffle3D(x, self.upscale_factor)

    def extra_repr(self):
        return 'upscale_factor={}'.format(self.upscale_factor)

class MyConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super(MyConv3d, self).__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(0, int((kernel_size-1)/2), int((kernel_size-1)/2)),
                              bias=bias)

    def forward(self, x):
        x = F.pad(x, pad=(0,)*4+(int((self.kernel_size-1)/2),)*2, mode='replicate')
        return self.conv(x)

if __name__ == '__main__':
    import torch
    aa = torch.randn(4, 3, 5, 16, 16)
    print(id(aa))
    aa_unshuffle = pixel_unshuffle3D(aa, 2)
    print(id(aa_unshuffle))
    print(aa_unshuffle.shape)
    bb = pixel_shuffle3D(aa_unshuffle, 2)
    print(id(bb))
    print(bb.shape)
    err = (aa - bb).abs().sum()
    print('Error: {:.2f}'.format(err))


