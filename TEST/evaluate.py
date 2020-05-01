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
from dataset import EvaluateDateset
from model import AE
from model import AE_D
from model_vc import AutoVC
from model_prosody import AE_D_Prosody
from model_vqvae import  VQVAE
from model_vqvae_attn import VQVAE_attn
from model_vqvae_prosody import VQVAE_Prosody
from model_mbv import MBV
from model_norm import MultipleNorm
# from model_norm2 import MultipleNorm
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

        # get data
        self.load_data()
        print('Load Data...')

        # init the model with config
        self.build_model()
        print('Build Model...')

        # load model
        self.load_model()
        print('Load Model...')

        # read speaker info
        self.read_speaker_info()
        print('Read SpeakerInfo...')

        # read filenames
        self.read_filenames()
        print('Read Filenames...')

        # sampled n speakers for evaluation
        self.sample_n_speakers()
        print('Sample speakers...')

    def load_data(self):
        if self.args.infer_random:
            self.dataset = h5py.File(os.path.join(self.args.data_dir, self.args.dataset), 'r')
        with open(os.path.join(self.args.data_dir, '{}_data.json'.format(self.args.val_set)), 'r') as f:
            self.indexes = json.load(f)
        return

    def build_model(self): 
        if self.args.model_type == 'AE':
            self.model = cc(AE(self.config))
        elif self.args.model_type == 'VQVAE':
            speaker_encoder = VoiceEncoder()
            self.model = cc(VQVAE(self.config, speaker_encoder))
        elif self.args.model_type == 'DAE':
            speaker_encoder = VoiceEncoder()
            self.model = cc(AE_D(self.config, speaker_encoder))
        elif self.args.model_type == 'AutoVC':
            speaker_encoder = VoiceEncoder()
            self.model = cc(AutoVC(32,256,512,32,speaker_encoder))
        elif self.args.model_type == 'Prosody':
            speaker_encoder = VoiceEncoder()
            self.model = cc(VQVAE_Prosody(self.config, speaker_encoder))
        elif self.args.model_type == 'MBV':
            speaker_encoder = VoiceEncoder()
            self.model = cc(MBV(self.config, speaker_encoder))
        elif self.args.model_type == 'NORM':
            speaker_encoder = VoiceEncoder()
            self.model = cc(MultipleNorm(self.config, speaker_encoder))
        elif self.args.model_type == 'Attn':
            speaker_encoder = VoiceEncoder()
            self.model = cc(VQVAE_attn(self.config, speaker_encoder))

        self.model.eval()

        if self.args.use_wavenet:
            from wavenet import build_model
            self.vocoder = cc(build_model())

        return

    def load_model(self):
        if self.args.model_type == 'AE':
            model_path = os.path.join('./model_ae', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'DAE':
            model_path = os.path.join('./model_dvae', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'VQVAE':
            model_path = os.path.join('./model_dvae_selfattn_vqvae_c16', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'AutoVC':
            model_path = os.path.join('./model_autovc', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'Prosody':
            model_path = os.path.join('./model_vqvae128_vae16', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'MBV':
            model_path = os.path.join('./model_dvae_pd_mbv', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'NORM':
            model_path = os.path.join('./model_dvae_pd_ponorm_2_6', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'Attn':
            model_path = os.path.join('./model_selfattn_vqvae64_vae8', 'model-{}.ckpt'.format(self.args.load_iteration))


        print('Load model from {}'.format(model_path))
        self.model.load_state_dict(torch.load(model_path))
        if self.args.use_wavenet:
            self.vocoder.load_state_dict(torch.load("./wavenet/checkpoint_step001000000_ema.pth")["state_dict"])
        return

    def read_speaker_info(self):
        speaker_infos = {}
        man, woman = [], []
        if args.val_set == 'out_test':
            val_speaker_info = 'test_speakers.txt'
        else:
            val_speaker_info = 'train_speakers.txt'

        with open(os.path.join(self.args.data_dir, val_speaker_info), 'r') as f:
            for i, line in enumerate(f):
                speaker_id = line.strip().split()[0]
                speaker_info = line.strip().split()[1:]
                if speaker_id in self.indexes:
                    speaker_infos[speaker_id] = speaker_info
                    if speaker_info[1] == 'F':
                        woman.append(speaker_id)
                    elif speaker_info[1] == 'M':
                        man.append(speaker_id)

        self.speaker_infos = speaker_infos
        self.speakers = list(speaker_infos.keys())
        self.speakers_man = man
        self.speakers_woman = woman

    def read_filenames(self):
        speaker2filenames = defaultdict(lambda : {})
        for path in sorted(glob.glob(os.path.join(self.args.vctk_dir, '*/*.wav'))):
            filename = path.strip().split('/')[-1]
            speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
            speaker2filenames[speaker_id][utt_id] = path

        self.speaker2filenames = speaker2filenames

    def sample_n_speakers(self):
        # random n speakers are sampled
        self.samples_man = random.sample(self.speakers_man,int(self.args.n_speakers/2))
        self.samples_woman = random.sample(self.speakers_woman, int(self.args.n_speakers/2))
        samples = []
        for m, w in zip(self.samples_man, self.samples_woman):
            samples.append(m)
            samples.append(w)
        self.samples = samples
        self.samples_index = {speaker:i for i, speaker in enumerate(self.samples)}
        return

    def plot_quantized_embeddings(self):
        output_image_path = os.path.join(self.args.output_image_path,self.args.val_set,self.args.load_iteration)
        os.makedirs(output_image_path,exist_ok=True)
        embs = self.model.get_vqvae_embeddings()

        _embedding = nn.Embedding(64, 128)
        _embedding.weight.data.uniform_(-1/64, 1/64)
        embs_init = _embedding.weight.data.cpu()
        # embs = torch.cat((embs,embs_init),dim=0)

        # print(embs[:,:5])
        '''
        print('UMAP...shape_{}'.format(embs.shape))
        proj = umap.UMAP(n_neighbors=5, min_dist=0.2, metric='cosine').fit_transform(embs)
        x_min, x_max = proj.min(0), proj.max(0)
        proj = (proj - x_min) / (x_max - x_min)
        plt.subplot(1,2,1)
        plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
        plt.title('UMAP-{}x{}'.format(embs.shape[0],embs.shape[1]))
        '''
        print('t-SNE...final {} | init {}'.format(embs.shape, embs_init.shape))
        proj = TSNE(n_components=2, init='pca', perplexity=50, random_state=self.args.seed).fit_transform(embs)
        x_min, x_max = proj.min(0), proj.max(0)
        proj = (proj - x_min) / (x_max - x_min)
        #plt.subplot(1,2,2)
        plt.scatter(proj[:,0], proj[:,1], alpha=0.3)
        plt.title('tSNE-{}x{}'.format(embs.shape[0],embs.shape[1]))
        plt.savefig(os.path.join(output_image_path,'vqvae.png'))



    def plot_speaker_embeddings(self):

        output_image_path = os.path.join(self.args.output_image_path,self.args.val_set,self.args.load_iteration)
        os.makedirs(output_image_path,exist_ok=True)

        speakers = []
        utts = []
        for speaker in self.samples:
            speakers += [speaker] * len(self.indexes[speaker])
            utts += self.indexes[speaker]

        dataset = EvaluateDateset(os.path.join(self.args.data_dir, self.args.dataset), 
                                  speakers, utts, segment_size=None, 
                                  load_spectrogram='dmel')

        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True, collate_fn=dataset.collate_fn)
        embs = []

        for data in dataloader:
            spec = cc(data['spectrogram'])
            length = data['len']
            emb = self.model.get_speaker_embeddings(spec, length)
            embs += emb.detach().cpu().numpy().tolist() 
            print('Evaluate: {}/{}'.format(len(embs),len(dataloader)*128),end='\r') 

        embs = np.array(embs)
        norms = np.sqrt(np.sum(embs ** 2, axis=1, keepdims=True))
        embs = embs / norms 

        # t-SNE
        print('\nt-SNE...')
        embs_2d = TSNE(n_components=2, init='pca', perplexity=50).fit_transform(embs)
        x_min, x_max = embs_2d.min(0), embs_2d.max(0)
        embs_2d = (embs_2d - x_min) / (x_max - x_min)

        # plot to figure
        female_cluster = [i for i, speaker in enumerate(speakers) if self.speaker_infos[speaker][1] == 'F']
        male_cluster = [i for i, speaker in enumerate(speakers) if self.speaker_infos[speaker][1] == 'M']
        colors = np.array([self.samples_index[speaker] for speaker in speakers])
        plt.scatter(embs_2d[female_cluster, 0], embs_2d[female_cluster, 1],  c=colors[female_cluster], marker='x') 
        plt.scatter(embs_2d[male_cluster, 0], embs_2d[male_cluster, 1], c=colors[male_cluster], marker='o') 
        plt.savefig(os.path.join(output_image_path,'speaker.png'))
        plt.clf()
        plt.cla()
        plt.close()
        return

    def plot_segment_embeddings(self):

        output_image_path = os.path.join(self.args.output_image_path,self.args.val_set,self.args.load_iteration)
        os.makedirs(output_image_path,exist_ok=True)

        speakers = []
        utts = []
        for speaker in self.samples:
            speakers += [speaker] * len(self.indexes[speaker])
            utts += self.indexes[speaker]

        dataset = EvaluateDateset(os.path.join(self.args.data_dir, self.args.dataset), speakers, utts, 
                                  segment_size=self.config['data_loader']['segment_size'], 
                                  load_spectrogram='dmel')

        dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)
        batchiter = infinite_iter(dataloader)

        embs = []
        speakers = []
        # run the model 
        while (len(embs) < self.args.n_segments):
            data = next(batchiter) 
            speakers += data['speaker']
            data = cc(data['spectrogram'].permute(0,2,1))
            emb = self.model.get_speaker_embeddings(data)
            embs += emb.detach().cpu().numpy().tolist()
            print('Evaluate: {}/{}'.format(len(embs),self.args.n_segments),end='\r') 

        embs = np.array(embs)
        norms = np.sqrt(np.sum(embs ** 2, axis=1, keepdims=True))
        embs = embs / norms 

        # t-SNE
        print('\nt-SNE...')
        embs_2d = TSNE(n_components=2, init='pca', perplexity=50).fit_transform(embs)
        x_min, x_max = embs_2d.min(0), embs_2d.max(0)
        embs_2d = (embs_2d - x_min) / (x_max - x_min)

        # plot to figure
        female_cluster = [i for i, speaker in enumerate(speakers) if self.speaker_infos[speaker][1] == 'F']
        male_cluster = [i for i, speaker in enumerate(speakers) if self.speaker_infos[speaker][1] == 'M']
        colors = np.array([self.samples_index[speaker] for speaker in speakers])
        plt.scatter(embs_2d[female_cluster, 0], embs_2d[female_cluster, 1],  c=colors[female_cluster], marker='x') 
        plt.scatter(embs_2d[male_cluster, 0], embs_2d[male_cluster, 1], c=colors[male_cluster], marker='o') 
        plt.savefig(os.path.join(output_image_path,'segment.png'))
        plt.clf()
        plt.cla()
        plt.close()
        return

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

    def wave_generate(self,spec,path):
        
        if self.args.use_wavenet:
            from wavenet import wavegen
            path = path + '_wv.wav'
            waveform = wavegen(self.vocoder, c=spec)   
            librosa.output.write_wav(path, waveform, sr=16000)
             
        elif self.args.use_griflin:
            path = path + '_gf.wav'
            spectrogram2wav(spec, self.hparams, path, use_mel=True)

    def pad_seq(self, x, base=8):
        len_out = int(base * ceil(float(x.shape[0])/base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

    def inference_one_utterance(self, x, x_p, xs_d, xt_d, len_pad, pic_path=None):

        if self.args.model_type == 'AutoVC':
            spec = []
            for i in range(ceil(len(x)/128)):
                spec_seg = self.model.inference(
                    x[i*128:(i+1)*128].unsqueeze(0).permute(0,2,1),
                    xs_d.unsqueeze(0),
                    xt_d.unsqueeze(0),
                    [len(xs_d)],
                    [len(xt_d)])
                spec.append(spec_seg)
            spec = torch.cat(spec,dim=2)
            spec = spec[0, 0, :, :].detach().cpu().numpy()
        elif self.args.model_type == 'Prosody' or  self.args.model_type == 'MBV' or args.model_type == 'NORM' or args.model_type=='DAE':
            spec = self.model.inference(
                x.unsqueeze(0).permute(0,2,1), 
                x_p.unsqueeze(0).permute(0,2,1),
                xt_d.unsqueeze(0), 
                [len(xt_d)])
            spec = spec.transpose(1, 2).squeeze(0)
            spec = spec.detach().cpu().numpy()          
        elif self.args.model_type == 'VQVAE' or self.args.model_type == 'Attn':
            spec, indices = self.model.inference(
                x.unsqueeze(0).permute(0,2,1), 
                x_p.unsqueeze(0).permute(0,2,1),
                xt_d.unsqueeze(0), 
                [len(xt_d)])
            spec = spec.transpose(1, 2).squeeze(0)
            spec = spec.detach().cpu().numpy()    
            if pic_path:
                indices = indices.squeeze().tolist()
                with open(pic_path,'w') as f:
                    for n, i in enumerate(indices):
                        if n % int(len(indices)/4) == 0:
                            f.write('\n')
                        f.write('{} '.format(i))
        else:
            spec, attn_w = self.model.inference(x.unsqueeze(0).permute(0,2,1), x_p.unsqueeze(0).permute(0,2,1), xt_d.unsqueeze(0), [len(xt_d)])
            spec = spec.transpose(1, 2).squeeze(0)
            spec = spec.detach().cpu().numpy()
            if pic_path:
                label_fontsize = 16
                plt.figure(figsize=(16,16))
                plt.imshow(attn_w.squeeze(0).permute(1,0).cpu().detach().numpy(), aspect="auto", origin="lower", interpolation=None)
                plt.xlabel("Source timestamp", fontsize=label_fontsize)
                plt.ylabel("Target timestamp", fontsize=label_fontsize)
                plt.colorbar()
                plt.savefig(pic_path)
                plt.close()

        if len_pad != 0:
            spec = spec[:-len_pad, :]

        # T, mels
        return spec




    def infer_random(self,gender=False):
        #using the first sample from in_test
        random.seed(self.args.seed)

        if gender:
            source_sp = random.sample(self.speakers_man, 1)[0]
            target_sp = random.sample(self.speakers_woman, 1)[0]
        else:
            source_sp, target_sp = random.sample(self.speakers,2)
        
        source_utt = random.sample(self.indexes[source_sp],1)[0]
        target_utt = random.sample(self.indexes[target_sp],1)[0]
                
        # source_sp = '252'
        # target_sp = '229'
        # source_utt = '072'
        # target_utt = '001'

        # source_sp = '226'
        # target_sp = '238'
        # source_utt = '001'
        # target_utt = '001'
        display = False

        output_image_path = os.path.join(self.args.output_image_path,self.args.val_set,self.args.load_iteration,
                's{}_{}---t{}_{}'.format(source_sp, source_utt, target_sp, target_utt))
        output_audio_path = os.path.join(self.args.output_audio_path,self.args.val_set,self.args.load_iteration,
                's{}_{}---t{}_{}'.format(source_sp, source_utt, target_sp, target_utt))

        os.makedirs(output_image_path,exist_ok=True)
        os.makedirs(output_audio_path,exist_ok=True)

        # source_spec_ori = self.dataset[f'{source_sp}/{source_utt}/mel'][:]
        # target_spec_ori = self.dataset[f'{target_sp}/{target_utt}/mel'][:]
        
        source_spec_ori,_,_,_,_ = wav2spectrogram(self.speaker2filenames[source_sp][source_utt],self.hparams,display=display,silence=24)
        target_spec_ori,_,_,_,_ = wav2spectrogram(self.speaker2filenames[target_sp][target_utt],self.hparams,display=display,silence=24)

        print('source time {} | target time {}'.format(source_spec_ori.shape, target_spec_ori.shape))
        
        
        b = 32 if self.args.model_type=='AutoVC' else 8
        source_spec, source_len_pad = self.pad_seq(source_spec_ori, base=b)
        target_spec, target_len_pad = self.pad_seq(target_spec_ori, base=b)

        source_spec = cc(torch.from_numpy(source_spec))
        target_spec = cc(torch.from_numpy(target_spec))

        if self.args.model_type=='AE':
            source_dspec = source_spec.permute(1,0)
            target_dspec = target_spec.permute(1,0)
        else:
            source_dspec = cc(torch.from_numpy(self.dataset[f'{source_sp}/{source_utt}/dmel'][:]))    
            target_dspec = cc(torch.from_numpy(self.dataset[f'{target_sp}/{target_utt}/dmel'][:]))

        wav2wav(self.speaker2filenames[source_sp][source_utt],self.hparams,os.path.join(output_audio_path,'src_ori.wav'))
        wav2wav(self.speaker2filenames[target_sp][target_utt],self.hparams,os.path.join(output_audio_path,'tar_ori.wav'))

        # x = wav2numpy(self.speaker2filenames[source_sp][source_utt],self.hparams)
        # y = wav2numpy(self.speaker2filenames[target_sp][target_utt],self.hparams)
        # plt.subplot(2, 2, 1)
        # plt.plot(x)
        # plt.title('Source')
        # plt.subplot(2, 2, 2)
        # plt.plot(y)
        # plt.title('Target')
        # res = dtw(x,y)
        # x_warping_path = res.get_warping_path(target="query")
        # plt.plot(x[x_warping_path],label="aligned x to y")
        # y_warping_path = res.get_warping_path(target="reference")
        # plt.plot(y[y_warping_path],label="aligned y to x")
        # plt.show()
        # input()

        # for s in ['phd_ch1.wav', 'phd_en3.wav', 'welcome_ch3.wav', 'welcome_en3.wav']:

        #     PATH = '/home/mlpjb04/Voice_Conversion/Corpus-fun/'
        #     output_image_path = os.path.join(PATH,self.args.val_set+'_'+s[:-4],self.args.load_iteration,'image')
        #     output_audio_path = os.path.join(PATH,self.args.val_set+'_'+s[:-4],self.args.load_iteration,'audio')
        #     os.makedirs(output_image_path,exist_ok=True)
        #     os.makedirs(output_audio_path,exist_ok=True)

        #     source = s
        #     target_sp, target_utt = '229', '001'

        #     parameters = Params('/home/mlpjb04/Voice_Conversion/h5py-VCTK/AutoVC_d/hp.json')
        #     source_spec_ori, _, _ = wav2spectrogram(os.path.join(PATH,source),parameters,display=False)
        #     target_spec_ori = self.dataset[f'{target_sp}/{target_utt}/mel'][:]

        #     target_dspec = cc(torch.from_numpy(self.dataset[f'{target_sp}/{target_utt}/dmel'][:]))

        #     wav = preprocess_wav(Path(os.path.join(PATH,source)))
        #     source_dspec = d_wav2spec(wav)
        #     source_dspec = cc(torch.from_numpy(source_dspec))

        #     b = 32 if self.args.model_type=='AutoVC' else 8
        #     source_spec, source_len_pad = self.pad_seq(source_spec_ori, base=b)
        #     target_spec, target_len_pad = self.pad_seq(target_spec_ori, base=b)

        #     source_spec = cc(torch.from_numpy(source_spec))
        #     target_spec = cc(torch.from_numpy(target_spec))

        # sync
        self.plot_spectrograms(source_spec_ori, os.path.join(output_image_path,'src_sync.png'))
        self.wave_generate(source_spec_ori, os.path.join(output_audio_path,'src_sync'))

        self.plot_spectrograms(target_spec_ori, os.path.join(output_image_path,'tar_sync.png'))
        self.wave_generate(target_spec_ori, os.path.join(output_audio_path,'tar_sync'))

        # reconstruction
        source_spec_rec = self.inference_one_utterance(source_spec, source_spec, source_dspec, source_dspec, source_len_pad,  pic_path=os.path.join(output_image_path,'s_indices.txt'))
        self.plot_spectrograms(source_spec_rec, os.path.join(output_image_path,'src_rec.png'))
        self.wave_generate(source_spec_rec, os.path.join(output_audio_path,'src_rec'))

        target_spec_rec = self.inference_one_utterance(target_spec, target_spec, target_dspec ,target_dspec, target_len_pad,  pic_path=os.path.join(output_image_path,'t_indices.txt'))
        self.plot_spectrograms(target_spec_rec, os.path.join(output_image_path,'tar_rec.png'))
        self.wave_generate(target_spec_rec, os.path.join(output_audio_path,'tar_rec'))

        criterion = nn.L1Loss()
        loss_src_rec = criterion(torch.from_numpy(source_spec_rec), torch.from_numpy(source_spec_ori))
        loss_trg_rec = criterion(torch.from_numpy(target_spec_rec), torch.from_numpy(target_spec_ori))
        print('Source Rec Loss: {} | Target Rec Loss: {}'.format(loss_src_rec, loss_trg_rec))
        
        # conversion speaker
        s2t_spec = self.inference_one_utterance(x=source_spec, x_p=source_spec, xs_d=source_dspec, xt_d=target_dspec, len_pad=source_len_pad)
        self.plot_spectrograms(s2t_spec, os.path.join(output_image_path,'s_s_t.png'))
        self.wave_generate(s2t_spec, os.path.join(output_audio_path,'s_s_t'))

        t2s_spec = self.inference_one_utterance(x=target_spec, x_p=target_spec, xs_d=target_dspec, xt_d=source_dspec, len_pad=target_len_pad)
        self.plot_spectrograms(t2s_spec, os.path.join(output_image_path,'t_t_s.png'))
        self.wave_generate(t2s_spec, os.path.join(output_audio_path,'t_t_s'))

        # conversion prosody
        if args.model_type == 'Prosody' or args.model_type == 'MBV' or args.model_type == 'NORM' or args.model_type == 'VQVAE' or args.model_type == 'Attn':
            source_len, target_len = len(source_spec), len(target_spec)
            max_len = max(len(source_spec), len(target_spec))
            source_spec = cc(torch.from_numpy(np.pad(source_spec.cpu().numpy(), ((0,max_len-len(source_spec)),(0,0)), 'constant')))
            target_spec = cc(torch.from_numpy(np.pad(target_spec.cpu().numpy(), ((0,max_len-len(target_spec)),(0,0)), 'constant')))

            s2t_spec = self.inference_one_utterance(x=source_spec, x_p=target_spec, xs_d=source_dspec, xt_d=source_dspec, len_pad=source_len_pad)
            s2t_spec = s2t_spec[:source_len]
            self.plot_spectrograms(s2t_spec, os.path.join(output_image_path,'s_t_s.png'))
            self.wave_generate(s2t_spec, os.path.join(output_audio_path,'s_t_s'))

            t2s_spec = self.inference_one_utterance(x=target_spec, x_p=source_spec, xs_d=target_dspec, xt_d=target_dspec, len_pad=target_len_pad)
            t2s_spec = t2s_spec[:target_len]
            self.plot_spectrograms(t2s_spec, os.path.join(output_image_path,'t_s_t.png'))
            self.wave_generate(t2s_spec, os.path.join(output_audio_path,'t_s_t'))
        
        print('Complete...')

        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--val_set', default='in_test', help='[train, in_test, out_test]')
    parser.add_argument('--model_type', default='AE', help='[VQVAE, AE, DAE, AutoVC, Prosody, MBV, NORM, Attn]')
    parser.add_argument('--load_iteration', help='[Give model iteration]')

    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', default='/home/mlpjb04/Voice_Conversion/h5py-VCTK/AutoVC_d')
    parser.add_argument('-vctk_dir', '-vctk', default='/home/mlpjb04/Voice_Conversion/Corpus-VCTK/wav48')

    parser.add_argument('-hparams', '-hps', default='hp.json')
    parser.add_argument('-dataset', '-ds', default='dataset.hdf5')

    parser.add_argument('-output_path', default='./test')   
    parser.add_argument('-output_image_path', default='./test/image')
    parser.add_argument('-output_audio_path', default='./test/audio')

    parser.add_argument('--plot_speakers', action='store_true')
    parser.add_argument('-n_speakers', default=20, type=int)

    parser.add_argument('--plot_segments', action='store_true')
    parser.add_argument('-n_segments',default=3000,type=int)

    parser.add_argument('--plot_quantized', action='store_true')
 
    parser.add_argument('--infer_random', action='store_true')

    parser.add_argument('--use_wavenet', action='store_true')
    parser.add_argument('--use_griflin', default=True, type=bool)

    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    args.output_image_path = './test/image_{}'.format(args.model_type)
    args.output_audio_path = './test/audio_{}'.format(args.model_type)

    # load config file 
    with open(args.config) as f:
        config = yaml.load(f)

    hparams = Params(os.path.join(args.data_dir, args.hparams))

    evaluator = Evaluator(config=config, args=args, hparams=hparams)
    os.makedirs(args.output_image_path,exist_ok=True)
    os.makedirs(args.output_audio_path,exist_ok=True)


    if args.plot_speakers:
        evaluator.plot_speaker_embeddings()

    if args.plot_segments:
        evaluator.plot_segment_embeddings()

    if args.plot_quantized:
        evaluator.plot_quantized_embeddings()

    if args.infer_random:
        evaluator.infer_random(gender=True)

