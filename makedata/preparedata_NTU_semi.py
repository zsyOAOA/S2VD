#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-09-23 10:23:45

import cv2
import argparse
import numpy as np
from pathlib import Path
from scipy.io import savemat
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument('--ntu_path_semi', type=str, default='./testsets/real_NTURain/',
                                 help="Path to save the original NTURain datasets, (default: None)")
parser.add_argument('--train_path', type=str, default='/ssd1t/NTURain/train_semi/',
                                help="Path to save the prepared training datasets, (default: None)")
parser.add_argument('--patch_size', type=int, default=64,
                                help="Path to save the prepared training datasets, (default: None)")
parser.add_argument('--batch_size', type=int, default=12,
                                help="Path to save the prepared training datasets, (default: None)")
args = parser.parse_args()

overlap = int(args.patch_size/4)
step_size = args.patch_size - overlap

rain_types = sorted([x.stem.split('_')[0] for x in Path(args.ntu_path_semi).glob('*_Rain')])

for current_type in rain_types:
    print('Processing Rain Type:{:s}...'.format(current_type))
    rain_dir = Path(args.ntu_path_semi) / (current_type + '_Rain')
    im_rain_path_list = sorted([x for x in rain_dir.glob('*.jpg')])
    for ii, im_rain_path in enumerate(im_rain_path_list):
        im_rain = cv2.imread(str(im_rain_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
        if ii == 0:
            rain_data_temp = im_rain[:, np.newaxis, ]
        else:
            rain_data_temp = np.concatenate((rain_data_temp, im_rain[:, np.newaxis, ]), axis=1)  #  c x n x h x w

    # crop data into patch
    c, num_frame, h, w = rain_data_temp.shape
    inds_h = list(range(0, h-args.patch_size, step_size)) + [h-args.patch_size,]
    inds_w = list(range(0, w-args.patch_size, step_size)) + [w-args.patch_size,]
    num_patch = len(inds_h) * len(inds_w)
    rain_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)

    iter_patch = 0
    for hh in inds_h:
        for ww in inds_w:
            rain_data[iter_patch, ] = rain_data_temp[:, :, hh:hh+args.patch_size, ww:ww+args.patch_size]
            iter_patch += 1

    for kk in range(ceil(num_patch / args.batch_size)):
        start = kk * args.batch_size
        end = min((kk+1)*args.batch_size, num_patch)
        rain_data_batch = rain_data[start:end,]
        save_path = Path(args.train_path) / (current_type+'_rain_'+str(kk+1)+'.mat')
        if save_path.exists():
            save_path.unlink()
        savemat(str(save_path), {'rain_data':rain_data_batch})


