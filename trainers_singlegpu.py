#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-08-29 16:25:28

import os
import sys
import cv2
import shutil
import random
import numpy as np
from math import ceil
from pathlib import Path
from scipy.io import loadmat, savemat
from networks.derain_net import DerainNet
from networks.generators import GeneratorState, GeneratorRain
from skimage import img_as_float32, img_as_ubyte
from utils import batch_PSNR, batch_SSIM, calculate_parameters

# pytorch package
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

torch.set_default_dtype(torch.float32)

class trainer:
    def __init__(self, args):
        '''
        :param args: options
        '''
        # setting random seed
        self.seed = args['seed']
        self.set_seed()

        # setting visible gpu
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in list(args['gpu_id']))

        # collect training data
        self.resume = args['resume']
        self.latent_size = args['latent_size']
        self.state_size = args['state_size']
        self.motion_size = args['motion_size']
        self.latent_dir_name = args['latent_dir_name']
        self.train_path = args['train_path']
        self.train_path_semi = args['train_path_semi']
        self.tidy_train_data()   # self.train_data_list

        # collect testing data
        self.test_path = args['test_path']
        self.test_path_semi = args['test_path_semi']
        self.tidy_test_data()  # self.test_data, self.test_gt, self.test_data_semi, c x n x h x w, float, torch

        # network settings
        self.patch_size = args['patch_size']
        self.feature_state = args['feature_state']
        self.feature_rain_G = args['feature_rain_G']
        self.n_resblocks = args['n_resblocks']
        self.feature_derain_D = args['feature_derain_D']

        # training settings
        self.rho = args['rho']
        self.tv_weight = args['tv_weight']
        self.epsilon2 = args['epsilon2']
        self.delta = args['delta']
        self.epochs = args['epochs']
        self.resume = args['resume']
        self.lr_D = args['lr_D']
        self.lr_GState = args['lr_GState']
        self.lr_GRain = args['lr_GRain']
        self.weight_decay_D = args['weight_decay_D']
        self.weight_decay_GRain = args['weight_decay_GRain']
        self.weight_decay_GState = args['weight_decay_GState']
        self.milestones = args['milestones']
        self.factor_lr = args['factor_lr']
        self.max_grad_norm_D = args['max_grad_norm_D']
        self.log_dir = args['log_dir']
        self.model_dir = args['model_dir']
        self.max_iter_EM = args['max_iter_EM']
        self.pretrain_derain = args['pretrain_derain']
        self.truncate = args['truncate']
        self.truncate_test = args['truncate_test']
        self.langevin_steps = args['langevin_steps']
        self.print_freq  = args['print_freq']

    def set_seed(self):
        print('*'*150)
        print('Setting random seed: {:d}...'.format(self.seed))
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tidy_train_data(self):
        print('*'*150)
        print('Making training data...')

        self.train_data_list = []
        make_latent = True if self.resume is None else False
        latent_dir = Path(self.train_path).parent / self.latent_dir_name
        if make_latent:
            if latent_dir.exists():
                shutil.rmtree(str(latent_dir))
            latent_dir.mkdir()

        # labelled data
        rain_data_list = sorted([str(x) for x in Path(self.train_path).glob('*_rain_*.mat')])
        for rain_path in rain_data_list:
            parts_path = Path(rain_path).name.split('_')
            gt_path = Path(rain_path).parent / (parts_path[0]+'_gt_'+parts_path[-1])
            latent_path = latent_dir / Path(rain_path).name.replace('rain', 'latent')
            state_path = latent_dir / Path(rain_path).name.replace('rain', 'state')
            motion_path = latent_dir / Path(rain_path).name.replace('rain', 'motion')
            generator_path = latent_dir / (Path(rain_path).stem.replace('rain', 'generator')+'.pt')

            if make_latent:
                rain_data = loadmat(rain_path)['rain_data']
                num_batch, _, num_frame = rain_data.shape[:3]
                Z = np.random.randn(num_batch, num_frame, self.latent_size).astype(np.float32)
                savemat(str(latent_path), {'Z': Z})

                S = np.random.randn(num_batch, self.state_size).astype(np.float32)
                savemat(str(state_path), {'S': S})

                M = np.random.randn(num_batch, self.motion_size).astype(np.float32)
                savemat(str(motion_path), {'M': M})

            self.train_data_list.append({'rainy': rain_path,
                                         'gt': str(gt_path),
                                         'generator': str(generator_path),
                                         'state': str(state_path),
                                         'motion': str(motion_path),
                                         'latent': str(latent_path)})

        # unlabelled data
        if self.train_path_semi:
            rain_data_semi_list = sorted([str(x) for x in Path(self.train_path_semi).glob('*_rain_*.mat')])
            for rain_path_semi in rain_data_semi_list:
                latent_path_semi = latent_dir / Path(rain_path_semi).name.replace('rain', 'latent')
                state_path_semi = latent_dir / Path(rain_path_semi).name.replace('rain', 'state')
                motion_path_semi = latent_dir / Path(rain_path_semi).name.replace('rain', 'motion')
                generator_path_semi = latent_dir / (Path(rain_path_semi).stem.replace('rain', 'generator')+'.pt')

                if make_latent:
                    rain_data_semi = loadmat(rain_path_semi)['rain_data']
                    num_batch, _, num_frame = rain_data_semi.shape[:3]
                    Z = np.random.randn(num_batch, num_frame, self.latent_size).astype(np.float32)
                    savemat(str(latent_path_semi), {'Z': Z})

                    S = np.random.randn(num_batch, self.state_size).astype(np.float32)
                    savemat(str(state_path_semi), {'S': S})

                    M = np.random.randn(num_batch, self.motion_size).astype(np.float32)
                    savemat(str(motion_path_semi), {'M': M})

                self.train_data_list.append({'rainy': rain_path_semi,
                                             'generator': str(generator_path_semi),
                                             'state': str(state_path_semi),
                                             'motion': str(motion_path_semi),
                                             'latent': str(latent_path_semi)})

        random.shuffle(self.train_data_list)

    def tidy_test_data(self):
        print('*'*150)
        print('Making testing data...')

        test_data_list = sorted([x for x in Path(self.test_path).glob('*.jpg')])
        for ii, rain_path in enumerate(test_data_list):
            gt_path = Path(self.test_path).parent / Path(self.test_path).stem.replace('Rain', 'GT') / rain_path.name
            im_rain = cv2.imread(str(rain_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
            im_gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
            if ii == 0:
                test_data = im_rain[:, np.newaxis,]
                test_gt = im_gt[:, np.newaxis,]
            else:
                test_data = np.concatenate((test_data, im_rain[:, np.newaxis,]), axis=1)
                test_gt = np.concatenate((test_gt, im_gt[:, np.newaxis,]), axis=1)

        self.test_gt = torch.from_numpy(img_as_float32(test_gt))
        self.test_data = torch.from_numpy(img_as_float32(test_data))

        test_data_semi_list = sorted([x for x in Path(self.test_path_semi).glob('*.jpg')])
        for ii, rain_path_semi in enumerate(test_data_semi_list):
            im_rain = cv2.imread(str(rain_path_semi), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
            if ii == 0:
                test_data_semi = im_rain[:, np.newaxis,]
            else:
                test_data_semi = np.concatenate([test_data_semi, im_rain[:, np.newaxis,]], axis=1)

        self.test_data_semi = torch.from_numpy(img_as_float32(test_data_semi))

    def build_network(self):
        self.GStateNet = GeneratorState(latent_size=self.latent_size,
                                        state_size=self.state_size,
                                        motion_size=self.motion_size,
                                        num_feature=self.feature_state).cuda()
        self.GRainNet = GeneratorRain(im_size=[self.patch_size,]*2,
                                 out_channels=3,
                                 state_size=self.state_size,
                                 num_feature=self.feature_rain_G).cuda()
        self.DNet = DerainNet(n_features=self.feature_derain_D,
                              n_resblocks=self.n_resblocks).cuda()
        print('*'*150)
        print('Number of parameters in Derain Net:{:d}'.format(calculate_parameters(self.DNet)))

    def decay_lr(self, ii):
        for stone in self.milestones:
            if (ii+1) == stone:
                self.optimizerD.param_groups[0]['lr'] *= self.factor_lr

    def load_checkpoint(self):
        if self.resume is not None:
            print('Loading checkpoint from {:s}'.format(self.resume))
            checkpoint_D = torch.load(self.resume)
            self.DNet.load_state_dict(checkpoint_D['DNet'])

            self.start_epoch = checkpoint_D['epoch']
            self.log_im_step = checkpoint_D['step_img']
            self.log_loss_step = checkpoint_D['step_loss']
            self.max_grad_norm_D = checkpoint_D['max_grad_norm_D']
            for ii in range(self.start_epoch):
                self.decay_lr(ii)
        else:
            self.start_epoch = 0
            self.log_loss_step = 0
            self.log_im_step = {'train':0, 'test':0}
            # path to save log
            if Path(self.log_dir).is_dir():
                shutil.rmtree(str(Path(self.log_dir)))
            Path(self.log_dir).mkdir()

            # path to save model
            if Path(self.model_dir).is_dir():
                shutil.rmtree(str(Path(self.model_dir)))
            Path(self.model_dir).mkdir()

    @staticmethod
    def load_data_video(current_path, semi=True):
        Y = loadmat(current_path['rainy'])['rain_data'] # num_batch x c x num_frame x p x p
        Z = loadmat(current_path['latent'])['Z']        # num_batch x num_frame x latent_size
        S = loadmat(current_path['state'])['S']         # num_batch x state_size
        M = loadmat(current_path['motion'])['M']        # num_batch x state_size
        if not semi:
            Y_gt = loadmat(current_path['gt'])['gt_data']   # num_batch x c x num_frame x p x p
            return Y, Y_gt, Z, S, M
        else:
            return Y, Z, S, M

    @staticmethod
    def tv1_norm3d(x, weight):
        '''
        Tv norm.
        :param x: B x 3 x num_frame x p x p
        :param weight: list with length 3
        '''
        B, C, N = x.shape[:3]
        x_tv = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().sum() * weight[0]
        y_tv = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().sum() * weight[1]
        z_tv = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().sum() * weight[2]
        tv_loss = (x_tv + y_tv + z_tv) / (B*C*N)
        return tv_loss

    def G_forward_truncate(self, truncate_Z, initial_state, motion_type):
        '''
        Forward propagation of Generator for truncated data.
        :param truncate_Z: Batch x num_frame x latent_size tensor
        :param initial_state:  Batch x state_size tensor
        :param motion_type:  Batch x state_size tensor
        '''
        rain_gen_all = []
        state_next = initial_state
        B, num_frame = truncate_Z.shape[:2]
        for ii in range(num_frame):
            input_Z = truncate_Z[:, ii, :].view([B,-1])
            state_next = self.GStateNet(input_Z, state_next, motion_type) # B x state_size
            rain_gen = self.GRainNet(state_next)             # B x 3 x p x p
            rain_gen_all.append(rain_gen)

        return torch.stack(rain_gen_all, dim=2), state_next

    def get_loss_MStep(self, Y, back_pre, rain_gen, gt):
        '''
        :param Y: B x 3 x num_frame x p x p tensor, rainy video
        :param back_pre: B x 3 x num_frame x p x p tensor, derained video
        :param rain_gen: B x 3 x num_frame x p x p tensor, generated rain
        :param gt: B x 3 x num_frame x p x p tensor, groundtruth video
        '''
        sigma = (Y - back_pre.detach() - rain_gen.detach()).flatten().std().item()
        likelihood = 0.5 / (sigma**2) * (Y - back_pre - rain_gen).square().mean()
        tv_loss = self.rho * self.tv1_norm3d(back_pre, self.tv_weight)
        if gt is None:
            mse_scale = torch.tensor(0)
        else:
            mse_scale = 0.5 / self.epsilon2 * (back_pre - gt).square().mean()
        loss = likelihood + mse_scale + tv_loss
        return loss, likelihood, mse_scale, tv_loss

    @staticmethod
    def get_loss_EStep(rain_gt, rain_gen):
        '''
        :param rain_gt: B x 3 x num_frame x p x p tensor, pesudoe rain layer groundtruth
        :param rain_gen: B x 3 x num_frame x p x p tensor, generated rain
        '''
        B, _, N = rain_gt.shape[:3]
        sigma = (rain_gt - rain_gen.detach()).flatten().std().item()
        loss = 0.5 / (sigma**2) * (rain_gt - rain_gen).square().sum()
        loss /= (B*N)
        return loss

    def freeze_Generator(self):
        for param in self.GStateNet.parameters():
            param.requires_grad = False
        for param in self.GRainNet.parameters():
            param.requires_grad = False

    def unfreeze_Generator(self):
        for param in self.GStateNet.parameters():
            param.requires_grad = True
        for param in self.GRainNet.parameters():
            param.requires_grad = True

    def predict_deraining(self):
        # Deraining
        self.DNet.eval()

        current_data_list = [self.test_data, self.test_data_semi] if self.train_path_semi else [self.test_data,]
        for kk, currrent_data in enumerate(current_data_list):
            num_frame = currrent_data.shape[1]
            test_data_derain = torch.zeros(currrent_data.shape)        # c x n x p x p
            for ii in range(ceil(num_frame / self.truncate_test)):
                start_ind = ii * self.truncate_test
                end_ind = min((ii+1) * self.truncate_test, num_frame)
                inputs = currrent_data[:, start_ind:end_ind,].cuda()  # c x truncate x p x p
                with torch.set_grad_enabled(False):
                    out = self.DNet(inputs.unsqueeze(0)).clamp_(0.0, 1.0).squeeze(0)
                test_data_derain[:, start_ind:end_ind, ] = out.cpu()

                if len(current_data_list) == 2 and kk == 1:
                    x1 = vutils.make_grid(inputs.permute([1,0,2,3]), normalize=True, scale_each=True)
                    self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                    x2 = vutils.make_grid(out.permute([1,0,2,3]), normalize=True, scale_each=True)
                    self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                    self.log_im_step['test'] += 1
                else:
                    if random.randint(1,10) == 1:
                        x1 = vutils.make_grid(inputs.permute([1,0,2,3]), normalize=True, scale_each=True)
                        self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                        x2 = vutils.make_grid(out.permute([1,0,2,3]), normalize=True, scale_each=True)
                        self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                        self.log_im_step['test'] += 1

            if kk == 0:
                self.psnrm = batch_PSNR(test_data_derain[:, 2:-2,].permute([1,0,2,3]),
                                         self.test_gt[:, 2:-2,].permute([1,0,2,3]), ycbcr=False)
                self.ssimm = batch_SSIM(test_data_derain[:, 2:-2,].permute([1,0,2,3]),
                                         self.test_gt[:, 2:-2,].permute([1,0,2,3]), ycbcr=False)

    def train(self):
        # build network
        self.build_network()

        # optimizer
        self.optimizerD = optim.Adam(self.DNet.parameters(),
                                     lr=self.lr_D,
                                     weight_decay=self.weight_decay_D,
                                     betas = (0.5, 0.999))

        # Loading from one specific checkpoint
        self.load_checkpoint()

        #open the tensorboard
        self.writer = SummaryWriter(str(Path(self.log_dir)))

        # begin training
        for ii in range(self.start_epoch, self.epochs):
            self.DNet.train()
            lossM_epoch = likelihood_epoch = mse_epoch = tv_epoch = 0
            mean_norm_grad_epoch_D = 0

            for jj, current_path in enumerate(self.train_data_list):
                if ii >= self.pretrain_derain:
                    checkpoint_path_G = current_path['generator']
                    checkpoint_G = torch.load(checkpoint_path_G)
                    self.GStateNet.load_state_dict(checkpoint_G['GState'])
                    self.GRainNet.load_state_dict(checkpoint_G['GRain'])

                optimizerG = optim.Adam([{'params': self.GStateNet.parameters(),
                                          'lr': self.lr_GState,
                                          'weight_decay': self.weight_decay_GState},
                                         {'params': self.GRainNet.parameters(),
                                          'lr': self.lr_GRain,
                                          'weight_decay': self.weight_decay_GRain}],
                                        betas = (0.5, 0.999))

                # load data
                if 'gt' in current_path:
                    Y, Y_gt, Z, S, M = self.load_data_video(current_path, semi=False)
                else:
                    Y, Z, S, M = self.load_data_video(current_path, semi=True)
                assert self.patch_size == Y.shape[-1]
                num_batch, _, num_frame = Y.shape[:3]
                lossM_batch = likelihood_batch = mse_batch = tv_batch = 0
                mean_norm_grad_batch_D = 0
                input_M = torch.from_numpy(M).cuda()
                for tt in range(ceil(num_frame / self.truncate)):
                    t_slice = slice(tt * self.truncate, min((tt+1)*self.truncate, num_frame))
                    inputs = torch.from_numpy(img_as_float32(Y[:, :, t_slice, ])).cuda()
                    if 'gt' in current_path:
                        gt = torch.from_numpy(img_as_float32(Y_gt[:, :, t_slice, ])).cuda()
                    else:
                        gt = None
                    input_Z = torch.from_numpy(Z[:, t_slice,]).cuda()
                    if tt == 0:
                        input_S = torch.from_numpy(S).cuda()
                    else:
                        input_S = torch.zeros_like(state_next, requires_grad=False).copy_(state_next.data)
                    # EM-algorithm
                    for _ in range(self.max_iter_EM):
                        # M-Step　
                        self.optimizerD.zero_grad()
                        optimizerG.zero_grad()
                        rain_gen_M, state_next = self.G_forward_truncate(input_Z, input_S, input_M)
                        back_pre = self.DNet(inputs)
                        lossM, likelihood, mse_scale, tv = self.get_loss_MStep(inputs, back_pre,
                                                                                     rain_gen_M, gt)
                        lossM.backward()
                        current_norm_grad_D = nn.utils.clip_grad_norm_(self.DNet.parameters(), self.max_grad_norm_D)
                        self.optimizerD.step()
                        if (ii+1) > self.pretrain_derain:
                            optimizerG.step()

                        # accumulate loss of M-Step
                        lossM_batch += lossM.item()
                        likelihood_batch += likelihood.item()
                        mse_batch += mse_scale.item()
                        tv_batch += tv.item()
                        mean_norm_grad_batch_D += current_norm_grad_D

                        # E-Step　
                        if (ii+1) > self.pretrain_derain:
                            self.freeze_Generator()
                            rain_gt = inputs - back_pre.detach()
                            for ss in range(self.langevin_steps):
                                input_Z.requires_grad = True
                                input_M.requires_grad = True
                                if tt == 0:
                                    input_S.requires_grad = True
                                rain_gen_E, state_next = self.G_forward_truncate(input_Z, input_S, input_M)
                                lossE = self.get_loss_EStep(rain_gt, rain_gen_E)
                                lossE.backward()

                                if tt == 0:
                                    input_S = input_S - 0.5 * (self.delta**2) * (input_S.grad + input_S/(num_batch*num_frame))
                                    if ss < (self.langevin_steps/3):
                                        input_S = input_S + self.delta * torch.randn_like(input_S)
                                    input_S.detach_()
                                input_Z = input_Z - 0.5 * (self.delta**2) * (input_Z.grad + input_Z/(num_batch*num_frame))
                                input_M = input_M - 0.5 * (self.delta**2) * (input_M.grad + input_M/(num_batch*num_frame))
                                if ss < (self.langevin_steps/3):
                                    input_Z = input_Z + self.delta * torch.randn_like(input_Z)
                                    input_M = input_M + self.delta * torch.randn_like(input_M)
                                input_Z.detach_()
                                input_M.detach_()
                            self.unfreeze_Generator()

                    # update Z_rank and S_rank
                    if (ii+1) > self.pretrain_derain:
                        Z[:, t_slice,] = input_Z.data.cpu().numpy()
                        M = input_M.data.cpu().numpy()
                        if tt == 0:
                            S = input_S.data.cpu().numpy()

                    # tensorboard
                    if random.randint(1,20)==1:
                        ind_batch = random.randint(0, rain_gen_M.shape[0]-1)
                        x1 = vutils.make_grid(inputs[ind_batch,].squeeze().permute([1,0,2,3]), normalize=False, scale_each=False)
                        self.writer.add_image('Train Rainy Image', x1, self.log_im_step['train'])
                        x3 = vutils.make_grid(back_pre[ind_batch,].squeeze().permute([1,0,2,3]).clamp_(0.0, 1.0), normalize=False, scale_each=False)
                        self.writer.add_image('Train Deained Image', x3, self.log_im_step['train'])
                        x4 = rain_gen_M[ind_batch,].squeeze().permute([1,0,2,3])
                        x5 = (inputs[ind_batch,]-back_pre[ind_batch,]).squeeze().permute([1,0,2,3]).clamp_(min=0)
                        temp = vutils.make_grid(torch.cat([x4, x5], dim=0), normalize=True, scale_each=True)
                        self.writer.add_image('Train Rains and Residual', temp, self.log_im_step['train'])
                        self.log_im_step['train'] += 1

                # save the updated latent variable
                if (ii+1) > self.pretrain_derain:
                    savemat(current_path['latent'], {'Z':Z})
                    savemat(current_path['state'],  {'S':S})
                    savemat(current_path['motion'], {'M':M})

                # calculate the mean loss of each video
                lossM_batch /= ((tt+1)*self.max_iter_EM)
                likelihood_batch /= ((tt+1)*self.max_iter_EM)
                mse_batch /= ((tt+1)*self.max_iter_EM)
                tv_batch /= ((tt+1)*self.max_iter_EM)
                mean_norm_grad_batch_D /= ((tt+1)*self.max_iter_EM)

                lossM_epoch += lossM_batch
                likelihood_epoch += likelihood_batch
                mse_epoch += mse_batch
                tv_epoch += tv_batch
                mean_norm_grad_epoch_D += mean_norm_grad_batch_D

                # print log
                if (jj+1) % self.print_freq==0:
                    self.writer.add_scalar('LossM_Batch', lossM_batch, self.log_loss_step)
                    self.log_loss_step += 1

                    lr_D = self.optimizerD.param_groups[0]['lr']
                    lr_GState = optimizerG.param_groups[0]['lr']
                    lr_GRain = optimizerG.param_groups[1]['lr']
                    log_str = 'M-Step: Epoch:{:03d}/{:03d}, Video:{:03d}/{:03d}, ' + \
                                    'LossM:{:.2e}({:.2e}/{:.2e}/{:.2e}), GradD:{:.2e}/{:.2e}, ' + \
                                                                        'lrSRD:{:.2e}/{:.2e}/{:.2e}'
                    print(log_str.format(ii+1, self.epochs, jj+1, len(self.train_data_list),
                         lossM_batch, likelihood_batch, mse_batch, tv_batch, mean_norm_grad_batch_D,
                                                   self.max_grad_norm_D, lr_GState, lr_GRain, lr_D))

                # save the rain generator
                if (ii+1) >= self.pretrain_derain:
                    torch.save({'GState': self.GStateNet.state_dict(),
                                'GRain': self.GRainNet.state_dict()}, current_path['generator'])

            # calculate the mean loss of each epoch
            lossM_epoch /= (jj+1)
            likelihood_epoch /= (jj+1)
            mse_epoch /= (jj+1)
            tv_epoch /= (jj+1)
            mean_norm_grad_epoch_D /= (jj+1)

            # print loss and testing
            print('-'*150)
            log_str = 'Train: Epoch:{:02d}/{:02d}, LossM:{:.2e} ({:.2e}/{:.2e}/{:.2e}), GradD:{:.2e}/{:.2e}'
            print(log_str.format(ii+1, self.epochs, lossM_epoch, likelihood_epoch, mse_epoch,
                                                 tv_epoch, mean_norm_grad_epoch_D, self.max_grad_norm_D))
            # testing
            self.predict_deraining()
            print('='*150)
            log_str = 'Test: Epoch:{:02d}/{:02d}, PSNR={:4.2f}, SSIM={:6.4f}'
            print(log_str.format(ii+1, self.epochs, self.psnrm, self.ssimm))
            print('='*150)

            # tensorboard
            self.writer.add_scalar('PSNR', self.psnrm, ii)
            self.writer.add_scalar('SSIM', self.ssimm, ii)
            self.writer.add_scalar('LossM_Epoch', lossM_epoch, ii)

            self.max_grad_norm_D = min(self.max_grad_norm_D, mean_norm_grad_epoch_D)

            # adjust learning rate
            self.decay_lr(ii)

            # save model
            model_prefix = 'model_'
            save_path_model = str(Path(self.model_dir) / (model_prefix+str(ii+1)))
            torch.save({'epoch': ii+1,
                        'step_loss': self.log_loss_step+1,
                        'step_img': {x:self.log_im_step[x]+1 for x in self.log_im_step.keys()},
                        'max_grad_norm_D': self.max_grad_norm_D,
                        'DNet': self.DNet.state_dict(),
                        'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)
            model_state_prefix = 'model_state_'
            save_path_model_state = str(Path(self.model_dir) / (model_state_prefix+str(ii+1)+'.pt'))
            torch.save(self.DNet.state_dict(), save_path_model_state)

        # close tensorboard
        self.writer.close()

