import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import numpy as np

from math import ceil
from functools import reduce
from torch.nn.utils import spectral_norm
from utils import cc
from torch.autograd import Variable

class DummyEncoder(object):
    def __init__(self, encoder):
        self.encoder = encoder

    def load(self, target_network):
        self.encoder.load_state_dict(target_network.state_dict())

    def __call__(self, x):
        return self.encoder(x)

def pad_layer(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size[0]
    if kernel_size % 2 == 0:
        pad = (kernel_size//2, kernel_size//2 - 1)
    else:
        pad = (kernel_size//2, kernel_size//2)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def pad_layer_2d(inp, layer, pad_type='reflect'):
    kernel_size = layer.kernel_size
    if kernel_size[0] % 2 == 0:
        pad_lr = [kernel_size[0]//2, kernel_size[0]//2 - 1]
    else:
        pad_lr = [kernel_size[0]//2, kernel_size[0]//2]
    if kernel_size[1] % 2 == 0:
        pad_ud = [kernel_size[1]//2, kernel_size[1]//2 - 1]
    else:
        pad_ud = [kernel_size[1]//2, kernel_size[1]//2]
    pad = tuple(pad_lr + pad_ud)
    # padding
    inp = F.pad(inp, 
            pad=pad,
            mode=pad_type)
    out = layer(inp)
    return out

def pixel_shuffle_1d(inp, scale_factor=2):
    batch_size, channels, in_width = inp.size()
    channels //= scale_factor
    out_width = in_width * scale_factor
    inp_view = inp.contiguous().view(batch_size, channels, scale_factor, in_width)
    shuffle_out = inp_view.permute(0, 1, 3, 2).contiguous()
    shuffle_out = shuffle_out.view(batch_size, channels, out_width)
    return shuffle_out

def upsample(x, scale_factor=2):
    x_up = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x_up

def flatten(x):
    out = x.contiguous().view(x.size(0), x.size(1) * x.size(2))
    return out

def concat_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, c_channels]
    cond = cond.unsqueeze(dim=2)
    cond = cond.expand(*cond.size()[:-1], x.size(-1))
    out = torch.cat([x, cond], dim=1)
    return out

def append_sp_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, x_channels * 2]
    p = cond.size(1) // 2
    mean, std = cond[:, :p], cond[:, p:]
    out = x * std.unsqueeze(dim=2) + mean.unsqueeze(dim=2)
    return out

def append_pd_cond(x, cond):
    # x = [batch_size, x_channels, length]
    # cond = [batch_size, x_channels * 2, length]
    p = cond.size(1) // 2
    mean, std = cond[:, :p, :], cond[:, p:, :]
    out = x * std + mean
    return out

def conv_bank(x, module_list, act, pad_type='reflect'):
    outs = []
    for layer in module_list:
        out = act(pad_layer(x, layer, pad_type))
        outs.append(out)
    out = torch.cat(outs + [x], dim=1)
    return out

def get_act(act):
    if act == 'relu':
        return nn.ReLU()
    elif act == 'lrelu':
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

class MLP(nn.Module):
    def __init__(self, c_in, c_h, n_blocks, act, sn):
        super(MLP, self).__init__()
        self.act = get_act(act)
        self.n_blocks = n_blocks
        f = spectral_norm if sn else lambda x: x
        self.in_dense_layer = f(nn.Linear(c_in, c_h))
        self.first_dense_layers = nn.ModuleList([f(nn.Linear(c_h, c_h)) for _ in range(n_blocks)])
        self.second_dense_layers = nn.ModuleList([f(nn.Linear(c_h, c_h)) for _ in range(n_blocks)])

    def forward(self, x):
        h = self.in_dense_layer(x)
        for l in range(self.n_blocks):
            y = self.first_dense_layers[l](h)
            y = self.act(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            h = h + y
        return h

class SpeakerEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size,
            bank_size, bank_scale, c_bank, 
            n_conv_blocks, n_dense_blocks, 
            subsample, act, dropout_rate):
        super(SpeakerEncoder, self).__init__()
        self.c_in = c_in
        self.c_h = c_h
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.n_conv_blocks = n_conv_blocks
        self.n_dense_blocks = n_dense_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.pooling_layer = nn.AdaptiveAvgPool1d(1)
        self.first_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.second_dense_layers = nn.ModuleList([nn.Linear(c_h, c_h) for _ in range(n_dense_blocks)])
        self.output_layer = nn.Linear(c_h, c_out)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def conv_blocks(self, inp):
        out = inp
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        return out

    def dense_blocks(self, inp):
        out = inp
        # dense layers
        for l in range(self.n_dense_blocks):
            y = self.first_dense_layers[l](out)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = self.second_dense_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            out = y + out
        return out

    def forward(self, x):

        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.act(out)
        # conv blocks
        out = self.conv_blocks(out)
        # avg pooling
        out = self.pooling_layer(out).squeeze(2)
        # dense blocks
        out = self.dense_blocks(out)
        out = self.output_layer(out)
        return out

class InstanceEncoder(nn.Module):
    def __init__(self, c_in=80, c_h=128, c_out=2, kernel_size=5,
            bank_size=8, bank_scale=1, c_bank=128, 
            n_conv_blocks=6, subsample=[1, 2, 1, 2, 1, 2], 
            act='relu', dropout_rate=0):
        super(InstanceEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.out_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        out = pad_layer(out, self.out_layer)
        return out

class LayerEncoder(nn.Module):
    def __init__(self, c_in=80, c_h=128, c_out=6, kernel_size=5,
            bank_size=8, bank_scale=1, c_bank=128, 
            n_conv_blocks=6, subsample=[1, 2, 1, 2, 1, 2], 
            act='relu', dropout_rate=0):
        super(LayerEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.out_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.norm_layer(out)
        norm_layer = nn.LayerNorm(normalized_shape=out.size()[1:],elementwise_affine=False)
        out = norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)
            norm_layer = nn.LayerNorm(normalized_shape=y.size()[1:],elementwise_affine=False)
            y = norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.norm_layer(y)
            norm_layer = nn.LayerNorm(normalized_shape=y.size()[1:],elementwise_affine=False)
            y = norm_layer(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        out = pad_layer(out, self.out_layer)
        return out

class PositionEncoder(nn.Module):
    def __init__(self, c_in=80, c_h=128, c_out=6, kernel_size=5,
            bank_size=8, bank_scale=1, c_bank=128, 
            n_conv_blocks=6, subsample=[1, 2, 1, 2, 1, 2], 
            act='relu', dropout_rate=0):
        super(PositionEncoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList(
                [nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.out_layer = nn.Conv1d(c_h, c_out, kernel_size=1)

        self.in_norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.ps_norm_layer = PositionNorm(c_h, affine=False)

        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer)
        out = self.in_norm_layer(out)
        out = self.ps_norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.in_norm_layer(y)
            y = self.ps_norm_layer(y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.in_norm_layer(y)
            y = self.ps_norm_layer(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        out = pad_layer(out, self.out_layer)
        return out

class PositionNorm(nn.Module):
    def __init__(self, input_size, affine=False, return_stats=False,eps=1e-5):
        super(PositionNorm, self).__init__()
        self.return_stats = return_stats
        self.input_size = input_size
        self.eps = eps
        self.affine = affine

        if affine:
            self.beta = nn.Parameter(torch.zeros(1, 1, *input_size))
            self.gamma = nn.Parameter(torch.ones(1, 1, *input_size))
        else:
            self.beta, self.gamma = None, None

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = (x.var(dim=1, keepdim=True) + self.eps).sqrt()
        x = (x - mean) / std
        if self.affine:
            x = x * self.gamma + self.beta

        if self.return_stats:
            return x, mean, std
        else:
            return x


class Decoder(nn.Module):
    def __init__(self, 
            c_in=6, c_cond=256, c_h=128, c_out=80, 
            kernel_size=5,
            n_conv_blocks=6, upsample=[2,1,2,1,2,1], act='relu', sn=False, dropout_rate=0):
        super(Decoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.upsample = upsample
        self.act = get_act(act)
        f = spectral_norm if sn else lambda x: x
        
        # z
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.first_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) \
            for _ in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size)) \
            for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.in_norm_layer = nn.InstanceNorm1d(c_h, affine=False)
        self.ps_norm_layer = PositionNorm(c_h, affine=False)

        # pd
        self.in_conv_layer_pd = f(nn.Conv1d(2, c_h * 2, kernel_size=1))
        self.first_conv_layers_pd = nn.ModuleList([f(nn.Conv1d(c_h * 2, c_h * 2, kernel_size=kernel_size)) \
            for _ in range(n_conv_blocks)])
        self.second_conv_layers_pd = nn.ModuleList([f(nn.Conv1d(c_h * 2, c_h * up * 2, kernel_size=kernel_size)) \
            for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.in_norm_layer_pd = nn.InstanceNorm1d(c_h, affine=False)

        # sp
        self.conv_affine_layers_sp = nn.ModuleList([f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks*2)])

        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z, pd_cond, sp_cond):
        out = pad_layer(z, self.in_conv_layer)
        #norm_layer = nn.LayerNorm(normalized_shape=out.size()[1:],elementwise_affine=False)
        #out = norm_layer(out)
        out = self.in_norm_layer(out)
        out = self.ps_norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)

        out_pd = pad_layer(pd_cond, self.in_conv_layer_pd)
        out_pd = self.in_norm_layer_pd(out_pd)
        out_pd = self.act(out_pd)
        out_pd = self.dropout_layer(out_pd)

        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y_pd = pad_layer(out_pd, self.first_conv_layers_pd[l])

            y = self.in_norm_layer(y)
            y = self.ps_norm_layer(y)
            y_pd = self.in_norm_layer_pd(y_pd)
            y = append_pd_cond(y, y_pd)
            y = self.in_norm_layer(y)
            y = append_sp_cond(y, self.conv_affine_layers_sp[l*2](sp_cond))

            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y_pd = pad_layer(y_pd, self.second_conv_layers_pd[l])

            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
                y_pd = pixel_shuffle_1d(y_pd, scale_factor=self.upsample[l])

            y = self.in_norm_layer(y)
            y = self.ps_norm_layer(y)
            y_pd = self.in_norm_layer_pd(y_pd)
            y = append_pd_cond(y, y_pd)
            y = self.in_norm_layer(y)
            y = append_sp_cond(y, self.conv_affine_layers_sp[l*2+1](sp_cond))

            y = self.act(y)
            y = self.dropout_layer(y)

            if self.upsample[l] > 1:
                out_pd = upsample(out_pd, scale_factor=self.upsample[l])
                out = y + upsample(out, scale_factor=self.upsample[l]) 
            else:
                out = y + out
        out = pad_layer(out, self.out_conv_layer)
        return out

class MultipleNorm(nn.Module):
    def __init__(self, config, speaker_encoder):
        super(MultipleNorm, self).__init__()
        self.speaker_encoder = speaker_encoder
        self.ProsodyEncoder = InstanceEncoder()
        #self.LayerEncoder = LayerEncoder()
        self.PhonemeEncoder = PositionEncoder()
        self.decoder = Decoder()

    def forward(self, x, x_s, x_s_len):
        s_emb = self.get_speaker_embeddings(x_s, x_s_len)
        c_emb = self.ProsodyEncoder(x)
        d_emb = self.PhonemeEncoder(x)
        dec = self.decoder(d_emb, c_emb, s_emb)
        return s_emb, dec

    def inference(self, x, x_p, x_s, x_s_len):
        x_s_emb = self.get_speaker_embeddings(x_s, x_s_len)
        x_d_emb = self.PhonemeEncoder(x)
        xp_c_emb = self.ProsodyEncoder(x_p)

        # x_d_emb = torch.zeros_like(x_d_emb).cuda()
        # xp_c_emb = torch.zeros_like(xp_c_emb).cuda()
        x_s_emb = torch.zeros_like(x_s_emb).cuda()

        dec = self.decoder(x_d_emb, xp_c_emb , x_s_emb)
        return dec

    def get_speaker_embeddings(self, x_d, x_d_len):
        emb = []
        for d, l in zip(x_d, x_d_len):
            emb.append(d[:l])
        emb = torch.cat(emb,dim=0)   
        emb = self.speaker_encoder.embed_dvector(emb, x_d_len)
        return emb
 