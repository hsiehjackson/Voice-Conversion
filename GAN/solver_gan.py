import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
import glob
import re

from model_GAN import AdaGAN_GEN, AdaGAN_DEC
from resemblyzer import VoiceEncoder
from dataset import TotalDataset
from torch.utils.data import DataLoader
from utils import *
from functools import reduce
from collections import defaultdict

import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
import librosa.display as lds
import random

class Solver_gan(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config
        #print(config)

        # args store other information
        self.args = args
        #print(self.args)

        # logger to use tensorboard
        self.logger = Logger(self.args.logdir)

        # get dataloader
        self.get_data_loaders()

        # init the model with config
        self.build_model()
        self.save_config()

        if args.load_model:
            self.load_model()

    def save_model(self, iteration):
        # save model and their optimizer
        model = {
        'G_model': self.G.state_dict(),
        'D_model': self.D.state_dict(),
        'G_opt': self.G_opt.state_dict(),
        'D_opt': self.D_opt.state_dict()
        }
        torch.save(model, os.path.join(self.args.store_model_path, f'model-{iteration+1}.ckpt'))

    def save_config(self):
        with open(f'{self.args.store_model_path}.config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}.args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def load_model(self):
        print(f'Load model from {self.args.load_model_path}')
        model = torch.load(self.args.load_model_path)
        self.G.load_state_dict(model['G_model'])
        self.D.load_state_dict(model['D_model'])
        self.G_opt.load_state_dict(model['G_opt'])
        self.D_opt.load_state_dict(model['D_opt'])
        return

    def get_data_loaders(self):
        data_dir = self.args.data_dir
        
        self.train_dataset = TotalDataset(
                os.path.join(data_dir, 'dataset.hdf5'), 
                os.path.join(data_dir, 'train_data.json'), 
                segment_size=self.config['data_loader']['segment_size'],
                load_mel=self.config['data_loader']['load_mel'])

        self.train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=32, 
                shuffle=self.config['data_loader']['shuffle'], 
                num_workers=100,
                pin_memory=True,
                collate_fn=self.train_dataset.collate_fn)

        self.train_iter = infinite_iter(self.train_loader)

        return

    def build_model(self): 
        # create model, discriminator, optimizers

        self.G = cc(AdaGAN_GEN(self.config))
        self.D = cc(AdaGAN_DEC(**self.config['Discriminator']))

        optimizer = self.config['optimizer']
        self.G_opt = torch.optim.Adam(
                self.G.parameters(), 
                lr=optimizer['g_lr'], 
                betas=(optimizer['beta1'], optimizer['beta2']), 
                amsgrad=optimizer['amsgrad'], 
                weight_decay=optimizer['weight_decay'])
        self.D_opt = torch.optim.Adam(
                self.D.parameters(), 
                lr=optimizer['d_lr'], 
                betas=(optimizer['beta1'], optimizer['beta2']), 
                amsgrad=optimizer['amsgrad'], 
                weight_decay=optimizer['weight_decay'])
        self.G.train()
        self.D.train()

        params = sum(p.numel() for p in self.G.parameters())
        print('Total G parameters: {}'.format(params))
        params = sum(p.numel() for p in self.D.parameters())
        print('Total D parameters: {}'.format(params))
        return
    
    def reset_grad(self):
        self.G_opt.zero_grad()
        self.D_opt.zero_grad()
    
    def gradient_penalty(self, real, fake):
        alpha = cc(torch.rand(real.size(0), 1, 1)).expand_as(real)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        d_interpolates = self.D(interpolates)
        fake = cc(torch.ones(d_interpolates.shape))
        dydx = torch.autograd.grad(outputs=d_interpolates,
                                   inputs=interpolates,
                                   grad_outputs=fake,
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]

        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
        return torch.mean((dydx_l2norm-1)**2)


    def dis_step(self):
        # # ---------------------
        # #  Train Discriminator
        # # ---------------------
        data = next(self.train_iter)
        data_src, data_tar = data['mel_src'], data['mel_tar']
        x_s  = cc(data_src).permute(0,2,1)
        x_t = cc(data_tar).permute(0,2,1)

        datas = [[x_s, x_t]]
        # datas = [[x_s, x_t], [x_t, x_s]]
        D_real = 0.
        D_fake = 0.
        for data in datas:
            src, tar = data[0], data[1]
            _, _, _, s2t = self.G(src, tar)
            real_pred = self.D(tar)
            fake_pred = self.D(s2t.detach())
            d_real_adv_loss = - torch.mean(real_pred)
            d_fake_adv_loss = torch.mean(fake_pred)
            d_gp_loss = self.gradient_penalty(tar, s2t) * self.config['lambda']['lambda_gp']
            d_loss = (d_real_adv_loss + d_fake_adv_loss + d_gp_loss)
            self.reset_grad() 
            d_loss.backward()
            self.D_opt.step()
            D_real += d_real_adv_loss.item()
            D_fake += d_fake_adv_loss.item()

        meta = {'D_real': D_real/len(datas),
                'D_fake': D_fake/len(datas),
                'D_gp': d_gp_loss/len(datas)}

        return meta


    def gen_step(self):
        # -----------------
        #  Train Generator
        # -----------------
        data = next(self.train_iter)
        data_src, data_tar = data['mel_src'], data['mel_tar']
        x_s  = cc(data_src).permute(0,2,1)
        x_t = cc(data_tar).permute(0,2,1)
        criterion = nn.L1Loss()
        datas = [[x_s, x_t]]
        # datas = [[x_s, x_t], [x_t, x_s]]
        G_adv, G_rec, G_content, G_style = 0., 0., 0., 0.
        for data in datas:
            src, tar = data[0], data[1]
            # Original-to-target domain.
            _, _, t1, s2t = self.G(src, tar)
            gen_pred = self.D(s2t)
            g_adv_loss = - torch.mean(gen_pred) * self.config['lambda']['lambda_adv']
            # Target-to-original domain.
            feat_s2t, feat_src, t2, src_rec = self.G(s2t, src)
            g_rec_loss = criterion(src_rec, src) * self.config['lambda']['lambda_rec']
            g_content_loss = criterion(feat_s2t, t1) * self.config['lambda']['lambda_content']
            g_style_loss = criterion(feat_src,t2) *  self.config['lambda']['lambda_style']
            # Backward and optimize.
            g_loss = (g_adv_loss +  g_rec_loss + g_content_loss + g_style_loss)
            self.reset_grad() 
            g_loss.backward()
            self.G_opt.step()
            G_adv += g_adv_loss.item()
            G_rec += g_rec_loss.item()
            G_content += g_content_loss.item()
            G_style += g_style_loss.item()

            # print(g_rec_loss)
            # input()            
            # import librosa.display as lds
            # from matplotlib import pyplot as plt
            # for conv, sr in zip(src_rec, src):
            #     data1 = conv.detach().cpu().numpy()
            #     data2 = sr.detach().cpu().numpy()
            #     print(data1.shape, data2.shape)
            #     plt.subplot(2,1,1)
            #     lds.specshow(data1, y_axis='mel', x_axis='time', cmap=plt.cm.Blues)
            #     plt.subplot(2,1,2)
            #     lds.specshow(data2, y_axis='mel', x_axis='time', cmap=plt.cm.Blues)
            #     plt.show()
            # input()

        meta = {'G_rec': G_rec/len(datas),
                'G_adv': G_adv/len(datas),
                'G_content': G_content/len(datas),
                'G_style': G_style/len(datas)}

        return meta
        

    def train(self, n_iterations, dis_steps):
        for iteration in range(n_iterations):

            for step in range(dis_steps):
                dis_meta = self.dis_step()
            gen_meta = self.gen_step()

            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/GAN_train', dis_meta, iteration)
                self.logger.scalars_summary(f'{self.args.tag}/GAN_train', gen_meta, iteration)

            
            slot_value = (iteration+1, n_iterations) + tuple([value for value in gen_meta.values()]) + tuple([value for value in dis_meta.values()])
            log = '[%06d/%06d] | Grec:%.3f | Gadv:%.3f | GC:%.3f | GS:%.3f | Dreal:%.3f | Dfake:%.3f | Dgp:%.3f    '
            print(log % slot_value, end='\r')
            
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()
        return

