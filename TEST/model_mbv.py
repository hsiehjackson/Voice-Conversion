import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import numpy as np

from layer import VectorQuantizer
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
    out = x + cond
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

class ContentEncoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, c_out_cont, c_out_disc, kernel_size,
            bank_size, bank_scale, c_bank, 
            n_conv_blocks, subsample, 
            act, dropout_rate):
        super(ContentEncoder, self).__init__()
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
        self.cont_layer = nn.Conv1d(c_h, c_out_cont, kernel_size=1)
        #self.disc_layer = nn.Conv1d(c_h, c_out_disc, kernel_size=1)
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
        cont = pad_layer(out, self.cont_layer)
        #disc = pad_layer(out, self.disc_layer)
        cont, disc = cont[:,:4,:], cont[:,4:,:]
        return cont, disc

class Decoder(nn.Module):
    def __init__(self, 
            c_in, c_cond, c_h, c_out, 
            kernel_size,
            n_conv_blocks, upsample, act, sn, dropout_rate):
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
        self.norm_layer = nn.InstanceNorm1d(c_h, affine=False)

        # pd
        self.in_conv_layer_pd = f(nn.Conv1d(4, c_h, kernel_size=1))
        self.first_conv_layers_pd = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) \
            for _ in range(n_conv_blocks)])
        self.second_conv_layers_pd = nn.ModuleList([f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size)) \
            for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.norm_layer_pd = nn.InstanceNorm1d(c_h, affine=False)

        # sp
        self.conv_affine_layers_sp = nn.ModuleList([f(nn.Linear(c_cond, c_h * 2)) for _ in range(n_conv_blocks*2)])

        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z, pd_cond, sp_cond):
        out = pad_layer(z, self.in_conv_layer)
        out = self.norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)

        out_pd = pad_layer(pd_cond, self.in_conv_layer_pd)
        #out_pd = self.norm_layer(out_pd)
        out_pd = self.act(out_pd)
        out_pd = self.dropout_layer(out_pd)

        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.norm_layer(y)

            y_pd = pad_layer(out_pd, self.first_conv_layers_pd[l])
            #y_pd = self.norm_layer(y_pd)

            y = append_pd_cond(y, y_pd)
            y = append_sp_cond(y, self.conv_affine_layers_sp[l*2](sp_cond))
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y_pd = pad_layer(y_pd, self.second_conv_layers_pd[l])

            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
                y_pd = pixel_shuffle_1d(y_pd, scale_factor=self.upsample[l])

            y = self.norm_layer(y)
            #y_pd = self.norm_layer(y_pd)
            y = append_pd_cond(y, y_pd)
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


def gumbel_softmax(logits, temperature=0.1):
    def _sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape, requires_grad=True)
        dist = -Variable(torch.log(-torch.log(U + eps) + eps))
        return dist.cuda() if torch.cuda.is_available() else dist

    def _gumbel_softmax_sample(logits, temperature):
        y = logits + _sample_gumbel(logits.size())
        return F.softmax(y / temperature, dim=-1)

    y = _gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y


class VectorGumbel(nn.Module):
    def __init__(self, embedding_dim, mode):
        super(VectorGumbel, self).__init__()
        self._embedding_dim = embedding_dim
        self._mode = mode

        if mode == 'continuous' or mode == 'onehot':
            self.linear = nn.Linear(embedding_dim, embedding_dim)
        elif mode == 'mbv':
            self.linear = nn.Linear(embedding_dim, embedding_dim*2)
        elif mode == 'mbvae':
            self.linear = nn.Linear(embedding_dim, embedding_dim*2)
            self.log_uniform = Variable(torch.log(torch.ones(1) / 2)).cuda()
        elif mode == 'vae':
            self.linear_mean = nn.Linear(embedding_dim, embedding_dim)
            self.linear_var = nn.Linear(embedding_dim, embedding_dim)

    def forward(self,inputs,var=None):
        
        if self._mode == 'continuous':
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            input_shape = inputs.shape
            flat_input = inputs.view(input_shape[0], -1, self._embedding_dim)
            out = self.linear(flat_input)
            out_act = F.leaky_relu(out, negative_slope=0.2)
            out_act = out_act.view(input_shape).permute(0, 3, 1, 2)
            return out_act
        elif self._mode == 'onehot':
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            input_shape = inputs.shape
            flat_input = inputs.view(input_shape[0], -1, self._embedding_dim)
            out_proj = self.linear(flat_input)
            out_act = gumbel_softmax(out_proj)
            out_act = out_act.view(input_shape).permute(0, 3, 1, 2)
            return out_act
        elif self._mode == 'mbv':
            inputs = inputs.permute(0, 2, 1).contiguous()
            input_shape = inputs.shape
            flat_input = inputs.view(input_shape[0], -1, self._embedding_dim)
            out_proj = self.linear(flat_input)
            out_proj = out_proj.view(out_proj.size(0), out_proj.size(1), self._embedding_dim, 2)
            out_act = gumbel_softmax(out_proj)[:, :, :, 0].squeeze().view(input_shape)
            return out_act

        elif self._mode == 'mbvae':
            # convert inputs from BCHW -> BHWC
            inputs = inputs.permute(0, 2, 3, 1).contiguous()
            input_shape = inputs.shape
            flat_input = inputs.view(input_shape[0], -1, self._embedding_dim)
            out_proj = self.linear(flat_input)
            out_proj = out_proj.view(out_proj.size(0), out_proj.size(1), self._embedding_dim, 2)
            out_act = gumbel_softmax(out_proj)[:, :, :, 0].squeeze().view(input_shape).permute(0, 3, 1, 2)
            
            log_proj = F.log_softmax(out_proj, dim=3)
            proj = torch.exp(log_proj)
            kl_loss = torch.mean(torch.sum(torch.sum(proj * (log_proj - self.log_uniform), dim=3),dim=2))
            return kl_loss, out_act

        elif self._mode == 'vae':
            mu = inputs.permute(0, 2, 3, 1).contiguous()
            var = var.permute(0, 2, 3, 1).contiguous()
            input_shape = mu.shape
            flat_mu = mu.view(input_shape[0], -1, self._embedding_dim)
            flat_var = var.view(input_shape[0], -1, self._embedding_dim)
            std = flat_var.mul(0.5).exp_()
            eps = Variable(torch.cuda.FloatTensor(std.size()).normal_())
            out_act = eps.mul(std).add_(flat_mu)
            out_act = out_act.view(input_shape).permute(0, 3, 1, 2)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + flat_var - flat_mu.pow(2) - flat_var.exp()))

            return kl_loss, out_act


class MBV(nn.Module):
    def __init__(self, config, speaker_encoder):
        super(MBV, self).__init__()
        self.speaker_encoder = speaker_encoder
        self.content_encoder = ContentEncoder(**config['ContentEncoder'])
        self.discrete_encoder = VectorGumbel(6, 'mbv')
        self.embedding = nn.Linear(6, 128)
        self.decoder = Decoder(**config['Decoder'])

    def forward(self, x, x_s, x_s_len):
        s_emb = self.get_speaker_embeddings(x_s, x_s_len)
        c_emb, d_code = self.content_encoder(x)
        d_code = self.discrete_encoder(d_code)

        d_emb = self.embedding(d_code).permute(0,2,1)
        dec = self.decoder(d_emb, c_emb, s_emb)
        return s_emb, dec

    def inference(self, x, x_p, x_s, x_s_len):
        x_s_emb = self.get_speaker_embeddings(x_s, x_s_len)
        x_c_emb, x_d_code = self.content_encoder(x)
        xp_c_emb, xp_d_code = self.content_encoder(x_p)

        x_d_code = self.discrete_encoder(x_d_code)
        x_d_emb = self.embedding(x_d_code).permute(0,2,1)

        #x_d_emb = torch.zeros_like(x_d_emb).cuda()
        xp_c_emb = torch.zeros_like(xp_c_emb).cuda()

        dec = self.decoder(x_d_emb, xp_c_emb , x_s_emb)
        return dec

    def get_speaker_embeddings(self, x_d, x_d_len):
        emb = []
        for d, l in zip(x_d, x_d_len):
            emb.append(d[:l])
        emb = torch.cat(emb,dim=0)   
        emb = self.speaker_encoder.embed_dvector(emb, x_d_len)
        return emb
