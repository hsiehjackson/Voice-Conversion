import os
import json
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset

from resemblyzer import preprocess_wav
from resemblyzer import audio
from resemblyzer.hparams import *
from pathlib import Path


class EvaluateDateset(Dataset):
    def __init__(self, h5_path, speakers, utterances, segment_size, load_spectrogram='dmel'):
        self.dataset = h5py.File(h5_path, 'r')
        self.speakers = speakers
        self.utterances = utterances
        self.segment_size = segment_size
        self.load_spectrogram = load_spectrogram

    def __getitem__(self,index):

        data_lin = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/lin']
        data_mel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/mel']
        data_dmel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/dmel']
        data_lin = data_lin[:]
        data_mel = data_mel[:]
        data_dmel = data_dmel[:]

        # min_level = np.exp(-100 / 20 * np.log(10))
        # data_dmel = 20 * np.log10(np.maximum(min_level, data_dmel))
        # data_dmel = np.power(10.0, data_dmel * 0.05)
        # import matplotlib.pyplot as plt
        # import librosa.display as lds
        # lds.specshow(data_dmel.reshape(-1,40).T, y_axis='mel', x_axis='time', cmap=plt.cm.Blues)
        # plt.colorbar(format='%+2.0f dB')
        # plt.title('Mel spectrogram')
        # plt.show()
        # input()



        if self.segment_size is not None:
            t = random.randint(0, len(data_lin) - self.segment_size)
            data_lin = data_lin[t:t+self.segment_size]
            data_mel = data_mel[t:t+self.segment_size]
        
        if self.load_spectrogram == 'mel':
            return {'speaker': self.speakers[index], 'spectrogram': data_mel}
        elif self.load_spectrogram == 'lin':
            return {'speaker': self.speakers[index], 'spectrogram': data_lin}
        elif self.load_spectrogram == 'dmel':
            return {'speaker': self.speakers[index], 'spectrogram': data_dmel}

    def __len__(self):
        return len(self.speakers)

    def collate_fn(self, datas):
        batch = {}
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['len'] = [len(data['spectrogram']) for data in datas]
        batch['spectrogram'] = torch.tensor([self.pad_to_len(data['spectrogram'], max(batch['len'])) for data in datas])
        return batch

    def pad_to_len(self, arr, padded_len):
        padded_len = int(padded_len)
        padding = np.expand_dims(np.zeros_like(arr[0]),0)
        if (len(arr) < padded_len):
            for i in range(padded_len-len(arr)):
                arr = np.concatenate((arr,padding),axis=0)
        return arr[:padded_len]






class TotalDataset(Dataset):
    def __init__(self, h5_path, index_path, segment_size=128, load_mel=False):
        self.dataset = h5py.File(h5_path, 'r')

        # Split with different speaker
        with open(index_path) as f:
            self.class_indexes = json.load(f)
        # All data
        self.total_indexes = []
        for speaker_id, utt_ids in self.class_indexes.items():
            self.total_indexes += [[speaker_id,utt_id] for utt_id in utt_ids]

        print('Total utterances: {}'.format(len(self.total_indexes)))

        self.segment_size = segment_size
        self.load_mel = load_mel

    def __getitem__(self, index):
        speaker_id, utt_id = self.total_indexes[index]
        
        data = {
            'speaker': speaker_id,  
            'lin': self.dataset[f'{speaker_id}/{utt_id}/lin'],
            'mel': self.dataset[f'{speaker_id}/{utt_id}/mel'],
            'dmel': self.dataset[f'{speaker_id}/{utt_id}/dmel'],
            # N * 160 * 40
        }
        
        if self.segment_size is not None:
            t = random.randint(0, len(data['lin']) - self.segment_size)
            data['lin'] = data['lin'][t:t+self.segment_size]
            data['mel'] = data['mel'][t:t+self.segment_size]

        return data

    def __len__(self):
        return len(self.total_indexes)

    def collate_fn(self, datas):
        batch = {}
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['lin'] = torch.tensor([data['lin'] for data in datas])
        batch['mel'] = torch.tensor([data['mel'] for data in datas])
        batch['dmel_len'] = [len(data['dmel']) for data in datas]
        batch['dmel'] = torch.tensor([self.pad_to_len(data['dmel'], max(batch['dmel_len'])) for data in datas])
        return batch

    def pad_to_len(self, arr, padded_len):
        padded_len = int(padded_len)
        padding = np.expand_dims(np.zeros_like(arr[0]),0)
        if (len(arr) < padded_len):
            for i in range(padded_len-len(arr)):
                arr = np.concatenate((arr,padding),axis=0)
        return arr[:padded_len]


