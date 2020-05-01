import warnings
warnings.filterwarnings("ignore")
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
from tqdm import tqdm

from argparse import ArgumentParser
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from math import ceil
from tacotron.utils import melspectrogram2wav, wav2wav

from dataset import EvaluateDateset, TransferDateset
from classify_dataset import ClassifyTestDateset
from model_VAEGAN import AdaVAEGAN, Descriminator, Decoder
from utils import *
import librosa.display as lds


class Evaluator(object):
    def __init__(self, config, args):
        # config store the value of hyperparameters, turn to attr by AttrDict
        random.seed(args.seed)

        self.config = config

        # args store other information
        self.args = args

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
        if self.args.infer_random or self.args.plot_variance:
            self.dataset = h5py.File(os.path.join(self.args.data_dir, self.args.dataset), 'r')
        with open(os.path.join(self.args.data_dir, '{}_data.json'.format(self.args.val_set)), 'r') as f:
            self.indexes = json.load(f)
        if self.args.val_set == 'out_test':
            for k, v in self.indexes.items():
                self.indexes[k] = v[:int(len(v)/5)]

        if self.args.infer_sproofing:
            assert self.args.val_set == 'in_test'
            self.speaker2id={} 
            for i, (speaker_id, utt_ids) in enumerate(self.indexes.items()):
                self.speaker2id[speaker_id] = i
        return

    def build_model(self): 
        if self.args.model_type == 'AdaVAEGAN':
            self.model = cc(AdaVAEGAN(self.config))
            self.patch_model = cc(Decoder(**self.config['Decoder']))
            self.patch_model.eval()
        elif self.args.model_type == 'AdaVAE':
            self.model = cc(AdaVAEGAN(self.config))
        else:
            print('No model')

        self.model.eval()
        if self.args.use_wavenet:
            from wavenet import build_model
            self.vocoder = cc(build_model())
        if self.args.infer_sproofing:
            from classify import SpeakerClassifier
            self.Speakerclassifier = cc(SpeakerClassifier(n_class=len(self.speaker2id)))
            self.Speakerclassifier.eval()

    def load_model(self):
        if self.args.model_type == 'AdaVAEGAN':
            model_path = '/home/mlpjb04/Voice_Conversion/hsieh_code_VAE_GAN/model_AdaVAE/model-200000.ckpt'
            patch_model_path = os.path.join('./model_patchVAE', 'model-{}.ckpt'.format(self.args.load_iteration))
            patch_model_dict = torch.load(patch_model_path)
            self.patch_model.load_state_dict(patch_model_dict['G_model'])
        elif self.args.model_type == 'AdaVAE':
            model_path = '/home/mlpjb04/Voice_Conversion/hsieh_code_VAE_GAN/model_AdaVAE/model-200000.ckpt'
        
        print('Load model from {}'.format(model_path))
        self.model.load_state_dict(torch.load(model_path))

        if self.args.use_wavenet:
            self.vocoder.load_state_dict(torch.load("./wavenet/checkpoint_step001000000_ema.pth")["state_dict"])
        if self.args.infer_sproofing:
            self.Speakerclassifier.load_state_dict(torch.load("/home/mlpjb04/Voice_Conversion/hsieh_code_VAE_GAN/model_Classifier_512_vctk/model-70000-99.83.ckpt")["model"])
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
        if self.args.infer_sproofing:
            self.samples_man = self.speakers_man
            self.samples_woman = self.speakers_woman
        else:
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
        # in_test
        # self.samples = ['252', '240', '237', '341', '274', '236', '272', '329', '271', '301']
        # out_test
        # self.samples = ['232', '305', '227', '238', '263', '339', '376', '318', '286', '312']
        for speaker in self.samples:
            speakers += [speaker] * len(self.indexes[speaker])
            utts += self.indexes[speaker]

        use_spec = 'dmel' if self.args.model_type == 'AdaVAEd' else 'mel'
        dataset = EvaluateDateset(os.path.join(self.args.data_dir, self.args.dataset), 
                                  speakers, utts, segment_size=None, 
                                  load_spectrogram='mel')

        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        embs = []

        for data in dataloader:
            spec = cc(data['spectrogram'])
            emb = self.model.get_speaker_embeddings(spec)
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
        plt.savefig(os.path.join(output_image_path,'speaker.png'))
        plt.clf()
        plt.cla()
        plt.close()
        return

    def plot_transfer_embeddings(self):

        output_image_path = os.path.join(self.args.output_image_path,self.args.val_set,self.args.load_iteration)
        os.makedirs(output_image_path,exist_ok=True)

        speakers = []
        utts = []
        # in_test
        # self.samples = ['252', '240', '237', '341', '274', '236', '272', '329', '271', '301']
        # out_test
        # self.samples = ['232', '305', '227', '238', '263', '339', '376', '318', '286', '312']
        for speaker in self.samples:
            speakers += [speaker] * len(self.indexes[speaker])
            utts += self.indexes[speaker]

        use_spec = 'dmel' if self.args.model_type == 'AdaVAEd' else 'mel'
        dataset = TransferDateset(os.path.join(self.args.data_dir, self.args.dataset), 
                                  speakers, utts, self.indexes, segment_size=None)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

        embs = []
        embs_tran = []

        for data in dataloader:
            spec_tar = cc(data['tar'])
            spec_tar_d = cc(data['tar_dmel'])
            spec_src = cc(data['src'])

            emb = self.model.get_speaker_embeddings(spec_tar)

            with torch.no_grad():
                mu, emb, spec_tran = self.model.patch(spec_src, spec_tar)
                
                if self.args.model_type == 'AdaVAEGAN':
                    spec_residual = self.patch_model(mu, emb)
                    spec_tran = spec_tran + spec_residual

                emb_tran = self.model.get_speaker_embeddings(spec_tran)

            embs += emb.detach().cpu().numpy().tolist() 
            embs_tran += emb_tran.detach().cpu().numpy().tolist() 

            print('Evaluate: {}/{}'.format(len(embs),len(dataloader)),end='\r') 

        embs_all = embs + embs_tran

        embs_all = np.array(embs_all)
        norms = np.sqrt(np.sum(embs_all ** 2, axis=1, keepdims=True))
        embs_all = embs_all / norms 

        # t-SNE
        print('\nt-SNE...')
        embs_2d = TSNE(n_components=2, init='pca', perplexity=50).fit_transform(embs_all)
        x_min, x_max = embs_2d.min(0), embs_2d.max(0)
        embs_2d = (embs_2d - x_min) / (x_max - x_min)

        embs_2d_src = embs_2d[:len(embs)]
        embs_2d_tran = embs_2d[len(embs):]
        # plot to figure
        female_cluster = [i for i, speaker in enumerate(speakers) if self.speaker_infos[speaker][1] == 'F']
        male_cluster = [i for i, speaker in enumerate(speakers) if self.speaker_infos[speaker][1] == 'M']
        colors = np.array([self.samples_index[speaker] for speaker in speakers])
        # plt.scatter(embs_2d_src[female_cluster, 0], embs_2d_src[female_cluster, 1],  c=colors[female_cluster], marker='x') 
        # plt.scatter(embs_2d_src[male_cluster, 0], embs_2d_src[male_cluster, 1], c=colors[male_cluster], marker='o') 
        plt.scatter(embs_2d_tran[female_cluster, 0], embs_2d_tran[female_cluster, 1],  c=colors[female_cluster], marker='x') 
        plt.scatter(embs_2d_tran[male_cluster, 0], embs_2d_tran[male_cluster, 1], c=colors[male_cluster], marker='o') 
        plt.savefig(os.path.join(output_image_path,'transfer.png'))
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
            melspectrogram2wav(spec, path)

    def pad_seq(self, x, base=8):
        len_out = int(base * ceil(float(x.shape[0])/base))
        len_pad = len_out - x.shape[0]
        assert len_pad >= 0
        return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

    def inference_one_utterance(self, x, xt_d, len_pad, pic_path=None):

        if self.args.model_type == 'AdaVAEd':
            spec = self.model.inference(
                x.unsqueeze(0).permute(0,2,1), 
                xt_d.unsqueeze(0))
            spec = spec.transpose(1, 2).squeeze(0)
            spec = spec.detach().cpu().numpy()          

        else:
            spec = self.model.inference(
                x.unsqueeze(0).permute(0,2,1), 
                xt_d.unsqueeze(0).permute(0,2,1))
            spec = spec.transpose(1, 2).squeeze(0)
            spec = spec.detach().cpu().numpy()    

        if len_pad != 0:
            spec = spec[:-len_pad, :]

        # T, mels
        return spec


    def plot_variance(self,srcgender='F', targender='M',n_samples=30):

        output_image_path = os.path.join(self.args.output_image_path,self.args.val_set,self.args.load_iteration)
        os.makedirs(output_image_path,exist_ok=True)

        source_sp = random.sample(self.speakers_man, 1)[0] if srcgender=='M' else random.sample(self.speakers_woman, 1)[0]
        target_sp = random.sample(self.speakers_man, 1)[0] if targender=='M' else random.sample(self.speakers_woman, 1)[0]

        source_utts = random.sample(self.indexes[source_sp],n_samples)
        target_utts = random.sample(self.indexes[target_sp],n_samples)
        comp_target_utts = random.sample(self.indexes[target_sp],n_samples)

        tar, conv_tar = [], []
        for src_utt, tar_utt, comp_tar_utt in tqdm(zip(source_utts, target_utts, comp_target_utts)):
            
            source_spec_ori = self.dataset[f'{source_sp}/{src_utt}/mel'][:]
            target_spec_ori = self.dataset[f'{target_sp}/{tar_utt}/mel'][:]
            comp_target_spec_ori = self.dataset[f'{target_sp}/{comp_tar_utt}/mel'][:]
            b = 32 if self.args.model_type=='AutoVC' else 8
            source_spec, source_len_pad = self.pad_seq(source_spec_ori, base=b)
            target_spec, target_len_pad = self.pad_seq(target_spec_ori, base=b)
            source_spec = cc(torch.from_numpy(source_spec))
            target_spec = cc(torch.from_numpy(target_spec))
            target_dspec = cc(torch.from_numpy(self.dataset[f'{target_sp}/{tar_utt}/dmel'][:])) if self.args.model_type=='AdaVAEd' else target_spec
            s2t_spec = self.inference_one_utterance(source_spec, target_dspec, len_pad=source_len_pad)
            tar.append(comp_target_spec_ori)
            conv_tar.append(s2t_spec)
        
        gv_tar = np.concatenate(tar).var(axis=0)
        gv_conv_tar = np.concatenate(conv_tar).var(axis=0)

        plt.plot(gv_conv_tar, color='y', label='converted')
        plt.plot(gv_tar, color='g', label='target speaker')
        plt.ylabel('gv', fontsize=12)
        plt.legend(loc='upper right', prop={'size': 11})
        plt.title('{}2{}'.format(srcgender, targender))
        plt.savefig(os.path.join(output_image_path,'gv_{}2{}.png'.format(srcgender, targender)))
        plt.clf()
        plt.cla()
        plt.close()

    def infer_sproofing(self):

        speakers = []
        utts = []

        for speaker in self.samples:
            speakers += [speaker] * len(self.indexes[speaker])
            utts += self.indexes[speaker]

        dataset = ClassifyTestDateset(os.path.join(self.args.data_dir, self.args.dataset), speakers, utts, self.indexes, self.speaker2id, segment_size=None)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        total_acc = 0.

        for data in tqdm(dataloader):
            spec_tar = cc(data['tar'])
            spec_tar_d = cc(data['tar_dmel'])
            spec_src = cc(data['src'])
            label = cc(data['label'])

            with torch.no_grad():
                mu, emb, spec_conv = self.model.patch(spec_src, spec_tar)
                if self.args.model_type == 'AdaVAEGAN':
                    spec_residual = self.patch_model(mu, emb)
                    spec_conv = spec_conv + spec_residual

                logits = self.Speakerclassifier(spec_conv)
                # logits = self.Speakerclassifier(use_spec)
                _, ind = torch.max(logits, dim=1)
                acc = torch.sum((ind == label).type(torch.FloatTensor)) / label.size(0)
                total_acc += acc  

        print('Speakers: {} | Accuracy: {:.2f}%'.format(len(dataloader), total_acc/len(dataloader)*100))

    def infer_random(self,gender=False):
        #using the first sample from in_test

        if gender:
            source_sp = random.sample(self.speakers_man, 1)[0]
            target_sp = random.sample(self.speakers_woman, 1)[0]
        else:
            source_sp, target_sp = random.sample(self.speakers,2)
        
        source_utt = random.sample(self.indexes[source_sp],1)[0]
        target_utt = random.sample(self.indexes[target_sp],1)[0]
                

        if self.args.val_set == 'out_test':
            source_sp = '237'
            target_sp = '225'
            source_utt = '007'
            target_utt = '008'
        # source_sp = '227'
        # target_sp = '312'
        # source_utt = '099'
        # target_utt = '285'
        display = False

        output_image_path = os.path.join(self.args.output_image_path,self.args.val_set,self.args.load_iteration,
                's{}_{}---t{}_{}'.format(source_sp, source_utt, target_sp, target_utt))
        output_audio_path = os.path.join(self.args.output_audio_path,self.args.val_set,self.args.load_iteration,
                's{}_{}---t{}_{}'.format(source_sp, source_utt, target_sp, target_utt))

        os.makedirs(output_image_path,exist_ok=True)
        os.makedirs(output_audio_path,exist_ok=True)
        
        source_spec_ori = self.dataset[f'{source_sp}/{source_utt}/mel'][:]
        target_spec_ori = self.dataset[f'{target_sp}/{target_utt}/mel'][:]

        print('source time {} | target time {}'.format(source_spec_ori.shape, target_spec_ori.shape))
        
        
        b = 32 if self.args.model_type=='AutoVC' else 8
        source_spec, source_len_pad = self.pad_seq(source_spec_ori, base=b)
        target_spec, target_len_pad = self.pad_seq(target_spec_ori, base=b)

        source_spec = cc(torch.from_numpy(source_spec))
        target_spec = cc(torch.from_numpy(target_spec))

        if self.args.model_type=='AdaVAEd':
            source_dspec = cc(torch.from_numpy(self.dataset[f'{source_sp}/{source_utt}/dmel'][:]))
            target_dspec = cc(torch.from_numpy(self.dataset[f'{target_sp}/{target_utt}/dmel'][:]))
        else:
            source_dspec = source_spec   
            target_dspec = target_spec

        wav2wav(self.speaker2filenames[source_sp][source_utt],os.path.join(output_audio_path,'src_ori.wav'))
        wav2wav(self.speaker2filenames[target_sp][target_utt],os.path.join(output_audio_path,'tar_ori.wav'))
        # sync
        self.plot_spectrograms(source_spec_ori, os.path.join(output_image_path,'src_sync.png'))
        self.wave_generate(source_spec_ori, os.path.join(output_audio_path,'src_sync'))

        self.plot_spectrograms(target_spec_ori, os.path.join(output_image_path,'tar_sync.png'))
        self.wave_generate(target_spec_ori, os.path.join(output_audio_path,'tar_sync'))

        # reconstruction
        source_spec_rec = self.inference_one_utterance(source_spec, source_dspec, source_len_pad)
        self.plot_spectrograms(source_spec_rec, os.path.join(output_image_path,'src_rec.png'))
        self.wave_generate(source_spec_rec, os.path.join(output_audio_path,'src_rec'))

        target_spec_rec = self.inference_one_utterance(target_spec, target_dspec ,target_len_pad)
        self.plot_spectrograms(target_spec_rec, os.path.join(output_image_path,'tar_rec.png'))
        self.wave_generate(target_spec_rec, os.path.join(output_audio_path,'tar_rec'))

        criterion = nn.L1Loss()
        loss_src_rec = criterion(torch.from_numpy(source_spec_rec), torch.from_numpy(source_spec_ori))
        loss_trg_rec = criterion(torch.from_numpy(target_spec_rec), torch.from_numpy(target_spec_ori))
        print('Source Rec Loss: {} | Target Rec Loss: {}'.format(loss_src_rec, loss_trg_rec))
        
        # conversion speaker
        s2t_spec = self.inference_one_utterance(source_spec, target_dspec, len_pad=source_len_pad)
        self.plot_spectrograms(s2t_spec, os.path.join(output_image_path,'s2t.png'))
        self.wave_generate(s2t_spec, os.path.join(output_audio_path,'s2t'))

        t2s_spec = self.inference_one_utterance(target_spec, source_dspec, len_pad=target_len_pad)
        self.plot_spectrograms(t2s_spec, os.path.join(output_image_path,'t2s.png'))
        self.wave_generate(t2s_spec, os.path.join(output_audio_path,'t2s'))
        
        print('Complete...')

        return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--val_set', default='in_test', help='[train, in_test, out_test]')
    parser.add_argument('--model_type', default='AdaVAE', help='[AdaVAE, AdaVAEd, AdaVAEavg, AdaAE]')
    parser.add_argument('--load_iteration', help='[Give model iteration]')

    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', default='/home/mlpjb04/Voice_Conversion/h5py-VCTK/AdaVAE_d')
    parser.add_argument('-vctk_dir', '-vctk', default='/home/mlpjb04/Voice_Conversion/Corpus-VCTK/wav48')

    parser.add_argument('-dataset', '-ds', default='dataset.hdf5')

    parser.add_argument('-output_path', default='./test')   
    parser.add_argument('-output_image_path', default='./test/image')
    parser.add_argument('-output_audio_path', default='./test/audio')

    parser.add_argument('--plot_speakers', action='store_true')
    parser.add_argument('--plot_transfers', action='store_true')
    parser.add_argument('--plot_variance', action='store_true')
    parser.add_argument('-n_speakers', default=20, type=int)

    parser.add_argument('--infer_sproofing', action='store_true')
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

    evaluator = Evaluator(config=config, args=args)
    os.makedirs(args.output_image_path,exist_ok=True)
    os.makedirs(args.output_audio_path,exist_ok=True)

    if args.plot_speakers:
        evaluator.plot_speaker_embeddings()

    if args.plot_transfers:
        evaluator.plot_transfer_embeddings()

    if args.plot_variance:
        evaluator.plot_variance()

    if args.infer_random:
        evaluator.infer_random(gender=True)

    if args.infer_sproofing:
        evaluator.infer_sproofing()

