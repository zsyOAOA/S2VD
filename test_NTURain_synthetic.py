#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-10-24 13:53:35

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from utils import batch_PSNR, batch_SSIM
from skimage import img_as_float32, img_as_ubyte
from networks.derain_net import DerainNet

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

# Build the network
print('Loading from {:s}'.format(str(Path('./model_states/derainer_rho05.pt'))))
model = DerainNet(n_features=32, n_resblocks=8).cuda()
state_dict = torch.load(str(Path('./model_states/derainer_rho05.pt')))
model.load_state_dict(state_dict)
model.eval()

# load data
base_data_path = Path('/home/oa/code/python/S2VD_git/testsets/synthetic_NTURain')
rain_types = sorted([x.stem.split('_')[0] for x in base_data_path.glob('*_Rain')])

truncate = 24
psnr_all_y = []
ssim_all_y = []
for current_type in rain_types:
    rain_dir = base_data_path / (current_type + '_Rain')
    im_rain_path_list = sorted([x for x in rain_dir.glob('*.jpg')])
    for ii, im_rain_path in enumerate(im_rain_path_list):
        im_gt_path = base_data_path / (current_type+'_GT') / im_rain_path.name
        im_rain = img_as_float32(cv2.imread(str(im_rain_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1])
        im_gt = img_as_float32(cv2.imread(str(im_gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1])
        if ii == 0:
            rain_data = torch.from_numpy(im_rain.transpose([2,0,1])).unsqueeze(0).unsqueeze(2) # 1 x c x 1 x h x w
            gt_data = torch.from_numpy(im_gt.transpose([2,0,1])[np.newaxis,])  # 1 x c x h x w
        else:
            temp = torch.from_numpy(im_rain.transpose([2,0,1])).unsqueeze(0).unsqueeze(2) # 1 x c x 1 x h x w
            rain_data = torch.cat((rain_data, temp), dim=2)  #  1 x c x n x h x w
            temp = torch.from_numpy(im_gt.transpose([2,0,1])[np.newaxis, ])
            gt_data = torch.cat((gt_data, temp), dim=0)             #  n x c x h x w

    num_frame = rain_data.shape[2]
    inds_start = list(range(0, num_frame, truncate))
    inds_end = list(range(truncate, num_frame, truncate)) + [num_frame,]
    assert len(inds_start) == len(inds_end)
    inds_ext_start = [0,] + [x-2 for x in inds_start[1:]]
    inds_ext_end = [x+2 for x in inds_end[:-1]] + [num_frame,]

    derain_data = torch.zeros_like(rain_data)
    with torch.set_grad_enabled(False):
        for ii in range(len(inds_start)):
            start_ext, end_ext, start, end = [x[ii] for x in [inds_ext_start, inds_ext_end, inds_start, inds_end]]
            inputs = rain_data[:, :, start_ext:end_ext, :, :].cuda()
            out_temp = model(inputs)
            if ii == 0:
                derain_data[0, :, start:end, ] = out_temp[:, :, :-2,].cpu().clamp_(0.0, 1.0)
            elif (ii+1) == len(inds_start):
                derain_data[0, :, start:end, ] = out_temp[:, :, 2:,].cpu().clamp_(0.0, 1.0)
            else:
                derain_data[0, :, start:end, ] = out_temp[:, :, 2:-2,].cpu().clamp_(0.0, 1.0)

    derain_data = derain_data[:, :, 2:-2,].squeeze(0).permute([1, 0, 2, 3])
    gt_data = gt_data[2:-2,]
    psnrm_y = batch_PSNR(derain_data, gt_data, ycbcr=True)
    psnr_all_y.append(psnrm_y)
    ssimm_y = batch_SSIM(derain_data, gt_data, ycbcr=True)
    ssim_all_y.append(ssimm_y)
    print('Type:{:s}, PSNR:{:5.2f}, SSIM:{:6.4f}'.format(current_type, psnrm_y, ssimm_y))

mean_psnr_y = sum(psnr_all_y) / len(rain_types)
mean_ssim_y = sum(ssim_all_y) / len(rain_types)
print('MPSNR:{:5.2f}, MSSIM:{:6.4f}'.format(mean_psnr_y, mean_ssim_y))
