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

from model_VAEd import AdaVAEd
from resemblyzer import VoiceEncoder
from dataset import TotalDataset_d
from torch.utils.data import DataLoader
from utils import *
from functools import reduce
from collections import defaultdict

import matplotlib as mpl
# mpl.use('Agg')
from matplotlib import pyplot as plt
import librosa.display as lds
import random

class Solver_vaed(object):
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
        torch.save(self.model.state_dict(), os.path.join(self.args.store_model_path, f'model-{iteration+1}.ckpt'))
        torch.save(self.opt.state_dict(), os.path.join(self.args.store_model_path, f'model-{iteration+1}.opt'))

    def save_config(self):
        with open(os.path.join(self.args.store_model_path, 'config_pre.yaml'), 'w') as f:
            yaml.dump(self.config, f)
        with open(os.path.join(self.args.store_model_path, 'args.yaml'), 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def load_model(self):
        print(f'Load model from {self.args.load_model_path}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}.ckpt'))
        self.opt.load_state_dict(torch.load(f'{self.args.load_model_path}.opt'))
        return


    def get_data_loaders(self):
        data_dir = self.args.data_dir
        
        self.train_dataset = TotalDataset_d(
                os.path.join(data_dir, 'dataset.hdf5'), 
                os.path.join(data_dir, 'train_data.json'), 
                segment_size=self.config['data_loader']['segment_size'],
                load_mel=self.config['data_loader']['load_mel'])

        self.train_loader = DataLoader(
                dataset=self.train_dataset, 
                collate_fn=self.train_dataset.collate_fn,
                batch_size=self.config['data_loader']['batch_size'], 
                shuffle=self.config['data_loader']['shuffle'], 
                num_workers=self.config['data_loader']['num_workers'],
                pin_memory=True)

        self.train_iter = infinite_iter(self.train_loader)

        return

    def build_model(self): 
        speaker_encoder = VoiceEncoder()
        self.model = cc(AdaVAEd(self.config, speaker_encoder))
        optimizer = self.config['optimizer']
        self.opt = torch.optim.Adam(
                self.model.parameters(), 
                lr=optimizer['lr'], 
                betas=(optimizer['beta1'], optimizer['beta2']), 
                amsgrad=optimizer['amsgrad'], 
                weight_decay=optimizer['weight_decay'])

        params = sum(p.numel() for p in self.model.parameters())
        print('Total parameters: {}'.format(params))
        return
    

    def ae_step(self, data, data_d, data_d_len, lambda_kl):
        x  = cc(data).permute(0,2,1)
        x_d = cc(data_d)
        mu, log_sigma, emb, dec = self.model(x, x_d, data_d_len)
        criterion = nn.L1Loss()
        loss_rec = criterion(dec, x)
        loss_kl = 0.5 * torch.mean(torch.exp(log_sigma) + mu ** 2 - 1 - log_sigma)
        loss = self.config['lambda']['lambda_rec'] * loss_rec + lambda_kl * loss_kl
        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                max_norm=self.config['optimizer']['grad_norm'])
        self.opt.step()
        meta = {'loss_rec': loss_rec.item(),
                'loss_kl': loss_kl.item(),
                'grad_norm': grad_norm}
        return meta
        

    def train(self, n_iterations):
        for iteration in range(n_iterations):
            if iteration >= self.config['annealing_iters']:
                lambda_kl = self.config['lambda']['lambda_kl']
            else:
                lambda_kl = self.config['lambda']['lambda_kl'] * (iteration + 1) / self.config['annealing_iters'] 
            data = next(self.train_iter)
            meta = self.ae_step(data['mel'], data['dmel'], data['dmel_len'], lambda_kl)

            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/de_train', meta, iteration)
            loss_rec = meta['loss_rec']
            loss_kl = meta['loss_kl']

            print(f'AdaVAE_d:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.4f}, '
                    f'loss_kl={loss_kl:.4f}, lambda={lambda_kl:.1e}     ', end='\r')
            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()
        return

