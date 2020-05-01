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
# mpl.use('Agg')
from matplotlib import pyplot as plt
from math import ceil

from audio import spectrogram2wav, wav2wav
from params import Params
from dataset import EvaluateDateset
from model import AE
from model import AE_D
from model_vc import AutoVC
from model_vqvae import  VQVAE
from resemblyzer import VoiceEncoder
from utils import *
import librosa.display as lds

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
            self.model = cc(VQVAE(self.config))
        elif self.args.model_type == 'DAE':
            speaker_encoder = VoiceEncoder()
            self.model = cc(AE_D(self.config, speaker_encoder))
        elif self.args.model_type == 'AutoVC':
            speaker_encoder = VoiceEncoder()
            self.model = cc(AutoVC(32,256,512,32,speaker_encoder))
        self.model.eval()
        return

    def load_model(self):
        if self.args.model_type == 'AE':
            model_path = os.path.join('./model', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'DAE':
            model_path = os.path.join('./model_dvae', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'VQVAE':
            model_path = os.path.join('./model_vqvae', 'model-{}.ckpt'.format(self.args.load_iteration))
        elif self.args.model_type == 'AutoVC':
            model_path = os.path.join('./model_autovc', 'model-{}.ckpt'.format(self.args.load_iteration))
        print('Load model from {}'.format(model_path))
        self.model.load_state_dict(torch.load(model_path))
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
        random.seed(42)
        self.samples_man = random.sample(self.speakers_man,int(self.args.n_speakers/2))
        self.samples_woman = random.sample(self.speakers_woman, int(self.args.n_speakers/2))
        samples = []
        for m, w in zip(self.samples_man, self.samples_woman):
            samples.append(m)
            samples.append(w)
        self.samples = samples
        self.samples_index = {speaker:i for i, speaker in enumerate(self.samples)}
        return


    def plot_speaker_embeddings(self):

        output_image_path = os.path.join(self.args.output_image_path,self.args.val_set,self.args.load_iteration)
        os.makedirs(output_image_path,exist_ok=True)

        speakers = []
        utts = []
        for speaker in self.samples:
            speakers += [speaker] * len(self.indexes[speaker])
            utts += self.indexes[speaker]

        dataset = EvaluateDateset(os.path.join(self.args.data_dir, self.args.dataset), 
                                  speakers, utts, segment_size=None, load_spectrogram='dmel')

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
        embs = []

        for data in dataloader:
            data = cc(data['spectrogram'])
            emb = self.model.get_speaker_embeddings(data,[data.size(1)])
            embs += emb.detach().cpu().numpy().tolist() 
            print('Evaluate: {}/{}'.format(len(embs),len(dataloader)),end='\r') 

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
        # plt.savefig(os.path.join(output_image_path,'speaker.png'))
        plt.savefig('speaker.png')
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
                                  dset=self.args.val_set, 
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
    
    def pad_seq(self, x, base=32):
        len_out = int(base * ceil(float(x.shape[0])/base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

    def inference_one_utterance(self, x, xs_d, xt_d, len_pad):

        if len(x) > 128:
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

        else:
            spec = self.model.inference(
                x.unsqueeze(0).permute(0,2,1),
                xs_d.unsqueeze(0),
                xt_d.unsqueeze(0),
                [len(xs_d)],
                [len(xt_d)])
        spec = spec[0, 0, :, :].detach().cpu().numpy()
        
        if len_pad == 0:
            spec = spec[:-len_pad, :]


        return spec

    def infer_random(self, gender=False):
        # using the first sample from in_test
        random.seed(42)

        if gender:
            source_sp = random.sample(self.speakers_man, 1)[0]
            target_sp = random.sample(self.speakers_woman, 1)[0]
        else:
            source_sp, target_sp = random.sample(self.speakers,2)
        
        source_utt = random.sample(self.indexes[source_sp],1)[0]
        target_utt = random.sample(self.indexes[target_sp],1)[0]


        output_image_path = os.path.join(self.args.output_image_path,self.args.val_set,self.args.load_iteration,
                's{}_{}---t{}_{}'.format(source_sp, source_utt, target_sp, target_utt))
        output_audio_path = os.path.join(self.args.output_audio_path,self.args.val_set,self.args.load_iteration,
                's{}_{}---t{}_{}'.format(source_sp, source_utt, target_sp, target_utt))

        os.makedirs(output_image_path,exist_ok=True)
        os.makedirs(output_audio_path,exist_ok=True)

        source_spec_ori = self.dataset[f'{source_sp}/{source_utt}/mel'][:]
        target_spec_ori = self.dataset[f'{target_sp}/{target_utt}/mel'][:]

        print('source time {} | target time {}'.format(source_spec_ori.shape, target_spec_ori.shape))
        
        source_spec, source_len_pad = self.pad_seq(source_spec_ori)
        target_spec, target_len_pad = self.pad_seq(target_spec_ori)
        source_spec = cc(torch.from_numpy(source_spec))
        target_spec = cc(torch.from_numpy(target_spec))
        target_dspec = cc(torch.from_numpy(self.dataset[f'{target_sp}/{target_utt}/dmel'][:]))
        source_dspec = cc(torch.from_numpy(self.dataset[f'{source_sp}/{source_utt}/dmel'][:]))    

        wav2wav(self.speaker2filenames[source_sp][source_utt],self.hparams,os.path.join(output_audio_path,'src_ori.wav'))
        wav2wav(self.speaker2filenames[target_sp][target_utt],self.hparams,os.path.join(output_audio_path,'tar_ori.wav'))

        self.plot_spectrograms(source_spec_ori, os.path.join(output_image_path,'src_sync.png'))
        spectrogram2wav(source_spec_ori, self.hparams, os.path.join(output_audio_path,'src_sync.wav'), use_mel=True)

        self.plot_spectrograms(target_spec_ori, os.path.join(output_image_path,'tar_sync.png'))
        spectrogram2wav(target_spec_ori, self.hparams, os.path.join(output_audio_path,'tar_sync.wav'), use_mel=True)


        s2t_spec = self.inference_one_utterance(source_spec, source_dspec, target_dspec, source_len_pad)
        self.plot_spectrograms(s2t_spec, os.path.join(output_image_path,'s2t.png'))
        spectrogram2wav(s2t_spec, self.hparams, os.path.join(output_audio_path,'s2t.wav'), use_mel=True)

        t2s_spec = self.inference_one_utterance(target_spec, target_dspec, source_dspec, target_len_pad)
        self.plot_spectrograms(t2s_spec, os.path.join(output_image_path,'t2s.png'))
        spectrogram2wav(t2s_spec, self.hparams, os.path.join(output_audio_path,'t2s.wav'), use_mel=True)
        
        # reconstruction
        source_spec_rec = self.inference_one_utterance(source_spec, source_dspec, source_dspec, source_len_pad)

        self.plot_spectrograms(source_spec_rec, os.path.join(output_image_path,'src_rec.png'))
        spectrogram2wav(source_spec_rec, self.hparams, os.path.join(output_audio_path,'src_rec.wav'), use_mel=True)

        target_spec_rec = self.inference_one_utterance(target_spec, target_dspec, target_dspec, target_len_pad)
        self.plot_spectrograms(target_spec_rec, os.path.join(output_image_path,'tar_rec.png'))
        spectrogram2wav(target_spec_rec, self.hparams, os.path.join(output_audio_path,'tar_rec.wav'), use_mel=True)

        criterion = nn.L1Loss()
        loss_src_rec = criterion(torch.from_numpy(source_spec_rec), torch.from_numpy(source_spec_ori))
        loss_trg_rec = criterion(torch.from_numpy(target_spec_rec), torch.from_numpy(target_spec_ori))
        print('Source Rec Loss: {} | Target Rec Loss: {}'.format(loss_src_rec, loss_trg_rec))

        
        print('Complete...')

        return



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--val_set', default='in_test', help='[train, in_test, out_test]')
    parser.add_argument('--model_type', default='AE', help='[VQVAE, AE, DAE, AutoVC]')
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
 
    parser.add_argument('--infer_random', action='store_true')

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

    if args.infer_random:
        evaluator.infer_random(gender=True)

