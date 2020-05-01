import sys
import os 
import yaml
import glob
import random
import json
import h5py
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from argparse import ArgumentParser
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from math import ceil

import librosa
from audio import spectrogram2wav, wav2wav, wav2spectrogram, wav2numpy
from make_dataset import d_wav2spec
from resemblyzer import preprocess_wav
from resemblyzer import audio
from resemblyzer.hparams import *
from pathlib import Path

from params import Params
from model import AE_D
from resemblyzer import VoiceEncoder
from utils import *
import librosa.display as lds
import umap
from dtwalign import dtw
import warnings
warnings.filterwarnings("ignore")

class Evaluator(object):
    def __init__(self, config, args, hparams):
        # config store the value of hyperparameters, turn to attr by AttrDict
        self.config = config

        # args store other information
        self.args = args

        # hyperparams for audio
        self.hparams = hparams

        # init the model with config
        self.build_model()
        print('Build Model...')

        # load model
        self.load_model()
        print('Load Model...')



    def build_model(self): 
        speaker_encoder = VoiceEncoder()
        self.model = cc(AE_D(self.config, speaker_encoder))
        self.model.eval()
        if self.args.use_wavenet:
            from wavenet import build_model
            self.vocoder = cc(build_model())

        return

    def load_model(self):

        model_path = os.path.join('./model_dvae', 'model-{}.ckpt'.format(self.args.load_iteration))
        print('Load model from {}'.format(model_path))
        self.model.load_state_dict(torch.load(model_path))
        if self.args.use_wavenet:
            self.vocoder.load_state_dict(torch.load("./wavenet/checkpoint_step001000000_ema.pth")["state_dict"])
        return

    def wave_generate(self,spec,path):
        
        if self.args.use_wavenet:
            from wavenet import wavegen
            waveform = wavegen(self.vocoder, c=spec)   
            librosa.output.write_wav(path, waveform, sr=16000)
             
        elif self.args.use_griflin:
            spectrogram2wav(spec, self.hparams, path, use_mel=True)

    def pad_seq(self, x, base=8):
        len_out = int(base * ceil(float(x.shape[0])/base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

    def inference_one_utterance(self, x, x_p, xs_d, xt_d, len_pad, pic_path=None):
        spec = self.model.inference(
            x.unsqueeze(0).permute(0,2,1), 
            x_p.unsqueeze(0).permute(0,2,1),
            xt_d.unsqueeze(0), 
            [len(xt_d)])
        spec = spec.transpose(1, 2).squeeze(0)
        spec = spec.detach().cpu().numpy()          

        if len_pad != 0:
            spec = spec[:-len_pad, :]

        return spec


    def plot_spectrograms(self, data, pic_path):
        data = data.T
        lds.specshow(data, y_axis='mel', x_axis='time', cmap=plt.cm.Blues)
        plt.title('Mel spectrogram')
        plt.xlabel('Time', fontsize=20)
        plt.ylabel('Frequency', fontsize=20)
        plt.savefig(pic_path)
        plt.clf()
        plt.cla()
        plt.close()
        return

    def infer_random(self):

        display = False

        data_dir = sorted(glob.glob('/home/mlpjb04/Voice_Conversion/hsieh_code_autoVC/hungyi/result/*/'))

        for i, folder in enumerate(data_dir):

            print('[{}/{}] {}'.format(i+1,len(data_dir),folder.strip().split('/')[-2]))
            output_audio_path = folder
            output_image_path = os.path.join(folder,'image')
            os.makedirs(output_image_path,exist_ok=True)
            source_path = os.path.join(folder,'source.wav')
            target_path = os.path.join(folder,'target.wav')
        
            source_spec_ori,_,_,_,_ = wav2spectrogram(source_path,self.hparams,display=display,silence=24)
            target_spec_ori,_,_,_,_ = wav2spectrogram(target_path,self.hparams,display=display,silence=24)

            # print('source time {} | target time {}'.format(source_spec_ori.shape, target_spec_ori.shape))
        
            source_spec, source_len_pad = self.pad_seq(source_spec_ori, base=8)
            target_spec, target_len_pad = self.pad_seq(target_spec_ori, base=8)

            source_spec = cc(torch.from_numpy(source_spec))
            target_spec = cc(torch.from_numpy(target_spec))

            source_dspec = d_wav2spec(preprocess_wav(Path(source_path)))
            source_dspec = cc(torch.from_numpy(source_dspec))
            target_dspec = d_wav2spec(preprocess_wav(Path(target_path)))
            target_dspec = cc(torch.from_numpy(target_dspec))
            

            srec_spec = self.inference_one_utterance(x=source_spec, x_p=source_spec, xs_d=source_dspec, xt_d=source_dspec, len_pad=source_len_pad)
            trec_spec = self.inference_one_utterance(x=target_spec, x_p=target_spec, xs_d=target_dspec, xt_d=target_dspec, len_pad=target_len_pad)
            # self.plot_spectrograms(srec_spec, os.path.join(output_image_path,'source.png'))
            # self.plot_spectrograms(trec_spec, os.path.join(output_image_path,'target.png'))
            
            # conversion speaker
            s2t_spec = self.inference_one_utterance(x=source_spec, x_p=source_spec, xs_d=source_dspec, xt_d=target_dspec, len_pad=source_len_pad)
            # self.plot_spectrograms(s2t_spec, os.path.join(output_image_path,'s2t.png'))
            self.wave_generate(s2t_spec, os.path.join(output_audio_path,'s2t_gf.wav'))

            t2s_spec = self.inference_one_utterance(x=target_spec, x_p=target_spec, xs_d=target_dspec, xt_d=source_dspec, len_pad=target_len_pad)
            # self.plot_spectrograms(t2s_spec, os.path.join(output_image_path,'t2s.png'))
            self.wave_generate(t2s_spec, os.path.join(output_audio_path,'t2s_gf.wav'))
        


        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--load_iteration', help='[Give model iteration]')
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', default='/home/mlpjb04/Voice_Conversion/h5py-VCTK/AutoVC_d')
    parser.add_argument('-hparams', '-hps', default='hp.json')
    parser.add_argument('--infer_random', action='store_true')

    parser.add_argument('--use_wavenet', action='store_true')
    parser.add_argument('--use_griflin', default=True, type=bool)
    args = parser.parse_args()

    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)

    hparams = Params(os.path.join(args.data_dir, args.hparams))

    evaluator = Evaluator(config=config, args=args, hparams=hparams)

    if args.infer_random:
        evaluator.infer_random()

