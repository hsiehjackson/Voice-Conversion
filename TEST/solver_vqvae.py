import torch
import numpy as np
import sys
import os 
import torch.nn as nn
import torch.nn.functional as F
import yaml
import pickle
from model_vqvae import  VQVAE
from resemblyzer import VoiceEncoder
from dataset import TotalDataset
from torch.utils.data import DataLoader
from utils import *
from functools import reduce
from collections import defaultdict

class Solver_vqvae(object):
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
        torch.save(self.model.state_dict(), f'{self.args.store_model_path}model-{iteration+1}.ckpt')
        torch.save(self.opt.state_dict(), f'{self.args.store_model_path}model-{iteration+1}.opt')

    def save_config(self):
        with open(f'{self.args.store_model_path}.config.yaml', 'w') as f:
            yaml.dump(self.config, f)
        with open(f'{self.args.store_model_path}.args.yaml', 'w') as f:
            yaml.dump(vars(self.args), f)
        return

    def load_model(self):
        print(f'Load model from {self.args.load_model_path}')
        self.model.load_state_dict(torch.load(f'{self.args.load_model_path}.ckpt'))
        self.opt.load_state_dict(torch.load(f'{self.args.load_model_path}.opt'))
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
                collate_fn=self.train_dataset.collate_fn,
                batch_size=self.config['data_loader']['batch_size'], 
                shuffle=self.config['data_loader']['shuffle'], 
                num_workers=self.config['data_loader']['num_workers'],
                pin_memory=True)

        self.train_iter = infinite_iter(self.train_loader)

        return

    def build_model(self): 
        # create model, discriminator, optimizers
        speaker_encoder = VoiceEncoder()
        self.model = cc(VQVAE(self.config, speaker_encoder))
        #print(self.model)
        optimizer = self.config['optimizer']
        self.opt = torch.optim.Adam(
                self.model.parameters(), 
                lr=optimizer['lr'], 
                betas=(optimizer['beta1'], optimizer['beta2']), 
                amsgrad=optimizer['amsgrad'], 
                weight_decay=optimizer['weight_decay'])
        #print(self.opt)
        return

    def ae_step(self, data, data_d, data_d_len):
        x  = cc(data).permute(0,2,1)
        x_d = cc(data_d)
        x_d_len = data_d_len

        loss_vq, emb, dec, perplexity, indices = self.model(x, x_d, x_d_len)
        criterion = nn.L1Loss()
        loss_rec = criterion(dec, x)
        loss = self.config['lambda']['lambda_rec'] * loss_rec + loss_vq
        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config['optimizer']['grad_norm'])
        self.opt.step()
        meta = {'loss_rec': loss_rec.item(),
                'loss_vq': loss_vq.item(),
                'perplexity': perplexity,
                'select': len(set(indices.squeeze().tolist())),
                'grad_norm': grad_norm}

        return meta

    def train(self, n_iterations):
        for iteration in range(n_iterations):


            data = next(self.train_iter)
            meta = self.ae_step(data['mel'], data['dmel'], data['dmel_len'])
            # add to logger
            if iteration % self.args.summary_steps == 0:
                self.logger.scalars_summary(f'{self.args.tag}/pd_vqvae_train', meta, iteration)

            loss_rec = meta['loss_rec']
            loss_vq = meta['loss_vq']
            perplexity = meta['perplexity']
            select = meta['select']
            all_select = self.config['VectorQuantizer']['num_embeddings']
            print(f'VQVAE:[{iteration + 1}/{n_iterations}], loss_rec={loss_rec:.3f}, '
                    f'loss_vq={loss_vq:.3f}, perplexity={perplexity:.3f}, select={select}/{all_select}     ', end='\r')

            if (iteration + 1) % self.args.save_steps == 0 or iteration + 1 == n_iterations:
                self.save_model(iteration=iteration)
                print()
        return

