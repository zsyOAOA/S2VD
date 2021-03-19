#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-10-24 13:53:35

import os
import torch
import numpy as np
from pathlib import Path
from scipy.io import savemat,loadmat
from skimage import img_as_float32, img_as_ubyte
from networks.derain_net import DerainNet

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(1)

model_path = str(Path('./model_states/derainer_rho05.pt'))
# Build the network
print('Loading from {:s}'.format(model_path))
model = DerainNet(n_features=32, n_resblocks=8).cuda()
state_dict = torch.load(model_path)
model.load_state_dict(state_dict)
model.eval()

data_path = './testsets/real_MSCSC/night.mat'
# load data
rain_data = loadmat(data_path)['input'].transpose([2,3,0,1])[np.newaxis,]  # 1 x 3 x n x h x w, uint8
rain_data = torch.from_numpy(img_as_float32(rain_data))

truncate = 24
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

save_path = Path(data_path).stem +'_derain.mat'
savemat(save_path, {'derain_data':img_as_ubyte(derain_data.squeeze(0).permute([2,3,0,1]).numpy()),
                    'rain_data':img_as_ubyte(rain_data.squeeze(0).permute([2,3,0,1]).numpy())})

