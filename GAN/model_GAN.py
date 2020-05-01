import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as ag
import numpy as np
from math import ceil
from torch.nn.utils import spectral_norm
from utils import cc


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



class Encoder(nn.Module):
    def __init__(self, c_in, c_h, c_out, kernel_size,
            bank_size, bank_scale, c_bank, 
            n_conv_blocks, subsample, 
            act, dropout_rate):
        super(Encoder, self).__init__()
        self.n_conv_blocks = n_conv_blocks
        self.subsample = subsample
        self.act = get_act(act)
        self.conv_bank = nn.ModuleList([nn.Conv1d(c_in, c_bank, kernel_size=k) for k in range(bank_scale, bank_size + 1, bank_scale)])
        in_channels = c_bank * (bank_size // bank_scale) + c_in
        self.in_conv_layer = nn.Conv1d(in_channels, c_h, kernel_size=1)
        self.in_norm_layer = nn.InstanceNorm1d(c_h)
        self.first_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size) for _ \
                in range(n_conv_blocks)])
        self.first_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList([nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub) 
            for sub, _ in zip(subsample, range(n_conv_blocks))])
        self.second_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ \
                in range(n_conv_blocks)])

        self.output_layer = nn.Conv1d(c_h, c_out, kernel_size=1)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        out = conv_bank(x, self.conv_bank, act=self.act)
        # dimension reduction layer
        out = pad_layer(out, self.in_conv_layer) # N, C, L
        out = self.in_norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.first_norm_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            y = pad_layer(y, self.second_conv_layers[l])
            y = self.second_norm_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)
            if self.subsample[l] > 1:
                out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
            out = y + out
        output = pad_layer(out, self.output_layer)
        return output

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
        self.in_conv_layer = f(nn.Conv1d(c_in, c_h, kernel_size=1))
        self.in_norm_layer = nn.InstanceNorm1d(c_h)
        self.first_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) for _ \
                in range(n_conv_blocks)])
        self.first_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ \
                in range(n_conv_blocks)])
        self.second_conv_layers = nn.ModuleList(\
                [f(nn.Conv1d(c_h, c_h * up, kernel_size=kernel_size)) \
                for _, up in zip(range(n_conv_blocks), self.upsample)])
        self.second_norm_layers = nn.ModuleList([nn.InstanceNorm1d(c_h) for _ \
                in range(n_conv_blocks)])
        self.out_conv_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
        self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, z):
        out = pad_layer(z, self.in_conv_layer)
        out = self.in_norm_layer(out)
        out = self.act(out)
        out = self.dropout_layer(out)
        # 128 128 16
        # convolution blocks
        for l in range(self.n_conv_blocks):
            y = pad_layer(out, self.first_conv_layers[l])
            y = self.first_norm_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)            
            y = pad_layer(y, self.second_conv_layers[l])
            if self.upsample[l] > 1:
                y = pixel_shuffle_1d(y, scale_factor=self.upsample[l])
            y = self.second_norm_layers[l](y)
            y = self.act(y)
            y = self.dropout_layer(y)

            if self.upsample[l] > 1:
                out = y + upsample(out, scale_factor=self.upsample[l]) 
            else:
                out = y + out
        out = pad_layer(out, self.out_conv_layer)
        return out


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 3)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    conv_feat = normalized_feat * style_std.expand(size) + style_mean.expand(size)
    return conv_feat


class AdaGAN_GEN(nn.Module):
    def __init__(self, config):
        super(AdaGAN_GEN, self).__init__()
        self.encoder = Encoder(**config['Encoder']) 
        self.decoder = Decoder(**config['Decoder'])

    def forward(self, source, target):
        source_feat = self.encoder(source)
        target_feat = self.encoder(target)
        t = adaptive_instance_normalization(source_feat, target_feat)
        dec = self.decoder(t)
        return source_feat, target_feat, t, dec

    def inference(self, source, target):
        source_feat = self.encoder(source)
        target_feat = self.encoder(target)
        t = adaptive_instance_normalization(source_feat, target_feat)
        dec = self.decoder(t)
        return dec

class AdaGAN_DEC(nn.Module):
    def __init__(self, act, dropout_rate, sn):
        super(AdaGAN_DEC, self).__init__()
        f = spectral_norm if sn else lambda x: x
        self.act = get_act(act)
        self.conv1 = f(nn.Conv2d(1, 32, kernel_size=(5,5), stride=2))
        self.conv2 = f(nn.Conv2d(32, 64, kernel_size=(5,5), stride=2))
        self.conv3 = f(nn.Conv2d(64, 128, kernel_size=(5,5), stride=2))
        self.conv4 = f(nn.Conv2d(128, 256, kernel_size=(5,5), stride=2))
        self.conv5 = f(nn.Conv2d(256, 256, kernel_size=(5,5), stride=2))
        self.conv6 = f(nn.Conv2d(256, 32, kernel_size=1))
        self.conv7 = f(nn.Conv2d(32, 1, kernel_size=(16, 4)))
        self.drop1 = nn.Dropout2d(p=dropout_rate)
        self.drop2 = nn.Dropout2d(p=dropout_rate)
        self.drop3 = nn.Dropout2d(p=dropout_rate)
        self.drop4 = nn.Dropout2d(p=dropout_rate)
        self.drop5 = nn.Dropout2d(p=dropout_rate)
        self.drop6 = nn.Dropout2d(p=dropout_rate)
        self.ins_norm1 = nn.InstanceNorm2d(self.conv1.out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(self.conv2.out_channels)
        self.ins_norm3 = nn.InstanceNorm2d(self.conv3.out_channels)
        self.ins_norm4 = nn.InstanceNorm2d(self.conv4.out_channels)
        self.ins_norm5 = nn.InstanceNorm2d(self.conv5.out_channels)
        self.ins_norm6 = nn.InstanceNorm2d(self.conv6.out_channels)

    def conv_block(self, x, conv_layer, after_layers):
        out = pad_layer_2d(x, conv_layer)
        out = self.act(out)
        for layer in after_layers:
            out = layer(out)
        return out 

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        out = self.conv_block(x, self.conv1, [self.ins_norm1, self.drop1])
        out = self.conv_block(out, self.conv2, [self.ins_norm2, self.drop2])
        out = self.conv_block(out, self.conv3, [self.ins_norm3, self.drop3])
        out = self.conv_block(out, self.conv4, [self.ins_norm4, self.drop4])
        out = self.conv_block(out, self.conv5, [self.ins_norm5, self.drop5])
        out = self.conv_block(out, self.conv6, [self.ins_norm6, self.drop6])
        # GAN output value
        val = self.conv7(out)
        val = val.view(val.size(0), -1)
        mean_value = torch.mean(val, dim=1)
        return mean_value

# class AdaGAN_DEC(nn.Module):
#     def __init__(self, c_in, c_h, c_out, kernel_size,
#             bank_size, bank_scale, c_bank, 
#             n_conv_blocks, subsample, 
#             act, dropout_rate, sn):
#         super(AdaGAN_DEC, self).__init__()
#         f = spectral_norm if sn else lambda x: x
#         self.last_size = 128 // 2 ** int(len(subsample) / 2) # 16
#         self.n_conv_blocks = n_conv_blocks
#         self.subsample = subsample
#         self.act = get_act(act)
#         self.conv_bank = nn.ModuleList([f(nn.Conv1d(c_in, c_bank, kernel_size=k)) for k in range(bank_scale, bank_size + 1, bank_scale)])
#         in_channels = c_bank * (bank_size // bank_scale) + c_in
#         self.in_conv_layer = f(nn.Conv1d(in_channels, c_h, kernel_size=1))
#         self.first_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size)) \
#             for _ in range(n_conv_blocks)])
#         self.second_conv_layers = nn.ModuleList([f(nn.Conv1d(c_h, c_h, kernel_size=kernel_size, stride=sub)) \
#             for sub, _ in zip(subsample, range(n_conv_blocks))])
#         self.output_layer = f(nn.Conv1d(c_h, c_out, kernel_size=1))
#         self.logit_layer = f(nn.Linear(self.last_size, 1))
#         self.dropout_layer = nn.Dropout(p=dropout_rate)


#     def forward(self, x):
#         out = conv_bank(x, self.conv_bank, act=self.act)
#         out = pad_layer(out, self.in_conv_layer)
#         # out = self.norm_layer(out)
#         out = self.act(out)
#         out = self.dropout_layer(out)
#         # convolution blocks
#         for l in range(self.n_conv_blocks):
#             y = pad_layer(out, self.first_conv_layers[l])
#             # y = self.norm_layer(y)
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             y = pad_layer(y, self.second_conv_layers[l])
#             # y = self.norm_layer(y)
#             y = self.act(y)
#             y = self.dropout_layer(y)
#             if self.subsample[l] > 1:
#                 out = F.avg_pool1d(out, kernel_size=self.subsample[l], ceil_mode=True)
#             out = y + out
#         out = pad_layer(out, self.output_layer) # b, 1, 16
#         out = out.view(-1, self.last_size)
#         logit = self.logit_layer(out)
#         return logit