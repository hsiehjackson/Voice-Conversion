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

from model_VAEGAN import AdaVAEGAN, Descriminator, Decoder
from dataset import TotalDatasetGAN
from torch.utils.data import DataLoader
from utils import *

import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
import librosa.display as lds
import random

class Solver_patchgan(object):
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
        'model': self.model.state_dict(),
        'G_model': self.G.state_dict(),
        'D_model': self.D.state_dict(),
        'G_opt': self.G_opt.state_dict(),
        'D_opt': self.D_opt.state_dict()
        }

        torch.save(model, os.path.join(self.args.store_model_path, f'model-{iteration+1}.ckpt'))


    def save_config(self):
        with open(f'{self.args.store_model_path}config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}args.yaml', 'w') as f:
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
        
        self.train_dataset = TotalDatasetGAN(
                os.path.join(data_dir, 'dataset.hdf5'), 
                os.path.join(data_dir, 'train_data.json'), 
                segment_size=self.config['data_loader']['segment_size'],
                load_mel=self.config['data_loader']['load_mel'])

        self.train_loader = DataLoader(
                dataset=self.train_dataset, 
                batch_size=self.config['data_loader']['batch_size'], 
                shuffle=self.config['data_loader']['shuffle'], 
                num_workers=self.config['data_loader']['num_workers'],
                pin_memory=True)

        self.train_iter = infinite_iter(self.train_loader)

        return

    def build_model(self): 
        # create model, discriminator, optimizers

        self.model = cc(AdaVAEGAN(self.config))
        self.model.load_state_dict(torch.load('/home/mlpjb04/Voice_Conversion/hsieh_code_VAE_GAN/model_AdaVAE/model-200000.ckpt'))
        self.model.eval()
        print('Load VAE model...')

        self.G = cc(Decoder(**self.config['Decoder']))
        self.D = cc(Descriminator(**self.config['Discriminator']))

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
        data = next(self.train_iter)
        x  = cc(data['src_mel']).permute(0,2,1)
        x_t = cc(data['tar_mel']).permute(0,2,1)
        mu, emb, x_dec_ori = self.model.patch(x, x_t)
        x_dec_residual = self.G(mu, emb)
        x_dec = x_dec_ori + x_dec_residual

        real_pred = self.D(x)
        fake_pred = self.D(x_dec)
        d_real_adv_loss = - torch.mean(real_pred)
        d_fake_adv_loss = torch.mean(fake_pred)
        d_gp_loss = self.gradient_penalty(x, x_dec) * self.config['lambda']['lambda_gp']
        d_loss = (d_real_adv_loss + d_fake_adv_loss + d_gp_loss)
        reset_grad([self.D])
        d_loss.backward()
        grad_clip([self.D], self.config['optimizer']['grad_norm'])
        self.D_opt.step()
        meta = {'D_real':  d_real_adv_loss.item(),
                'D_fake': d_fake_adv_loss.item(),
                'D_gp': d_gp_loss.item() / self.config['lambda']['lambda_gp']}

        return meta

    def gen_step(self, lambda_kl):
        data = next(self.train_iter)
        x  = cc(data['src_mel']).permute(0,2,1)
        x_t = cc(data['tar_mel']).permute(0,2,1)
        mu, emb, x_dec_s2t = self.model.patch(x, x_t)
        x_dec_s2t_residual = self.G(mu, emb)
        x_dec_s2t = x_dec_s2t + x_dec_s2t_residual
        fake_pred = self.D(x_dec_s2t)
        g_loss_adv = - torch.mean(fake_pred) * self.config['lambda']['lambda_adv']

        mu, emb, x_dec_t = self.model.patch(x_t, x_t)
        x_dec_t_residual = self.G(mu, emb)
        x_dec_t = x_dec_t + x_dec_t_residual
        criterion = nn.L1Loss()
        g_loss_rec = criterion(x_dec_t, x_t) * self.config['lambda']['lambda_rec']

        g_loss = g_loss_rec + g_loss_adv 
        reset_grad([self.G])
        g_loss.backward()
        grad_clip([self.G], self.config['optimizer']['grad_norm'])
        self.G_opt.step()


        meta = {'G_rec': g_loss_rec.item() / self.config['lambda']['lambda_rec'],
                'G_adv': g_loss_adv.item() / self.config['lambda']['lambda_adv']}

        return meta
    

    def train(self, n_iterations, dis_steps=1):
        import time
        start = time.time()
        
        for iteration in range(n_iterations):


            if iteration >= self.config['annealing_iters']:
                lambda_kl = self.config['lambda']['lambda_kl']
            else:
                lambda_kl = self.config['lambda']['lambda_kl'] * (iteration + 1) / self.config['annealing_iters'] 

            for step in range(dis_steps):
                dis_meta = self.dis_step()

            gen_meta = self.gen_step(lambda_kl)


            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/PatchGAN_train', dis_meta, iteration)
                self.logger.scalars_summary(f'{self.args.tag}/PatchGAN_train', gen_meta, iteration)

            
            slot_value = (iteration+1, n_iterations) + tuple([value for value in gen_meta.values()]) + tuple([value for value in dis_meta.values()])
            log = '[%06d/%06d] | Grec:%.3f | Gadv:%.3f | Dreal:%.3f | Dfake:%.3f | Dgp:%.3f    '
            print(log % slot_value, end='\r')
            
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()
        return



