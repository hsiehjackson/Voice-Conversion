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
from augment import spec_augment

class EvaluateDateset(Dataset):
    def __init__(self, h5_path, speakers, utterances, segment_size, load_spectrogram='mel'):
        self.dataset = None
        self.h5_path = h5_path
        self.speakers = speakers
        self.utterances = utterances
        self.segment_size = segment_size
        self.load_spectrogram = load_spectrogram

    def __getitem__(self,index):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')

        data_lin = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/lin'][:]
        data_mel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/mel'][:].transpose(1,0)
        data_dmel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/dmel'][:]

        if self.segment_size is not None:
            t = random.randint(0, len(data_lin) - self.segment_size)
            data_lin = data_lin[t:t+self.segment_size]
            data_mel = data_mel[t:t+self.segment_size]
        

        if self.load_spectrogram == 'mel':
            return {'speaker': self.speakers[index], 'spectrogram': data_mel}
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


class TransferDateset(Dataset):
    def __init__(self, h5_path, speakers, utterances, indexes, segment_size):
        self.dataset = None
        self.h5_path = h5_path
        self.speakers = speakers
        self.utterances = utterances
        self.segment_size = segment_size
        self.indexes = indexes

    def __getitem__(self,index):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')

        data_lin = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/lin'][:]
        data_mel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/mel'][:].transpose(1,0)
        data_dmel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/dmel'][:]

        random_speaker =  random.sample(set(list(self.indexes.keys())) - set(self.speakers[index]), 1)[0]
        random_utt = random.sample(self.indexes[random_speaker],1)[0]
        random_mel = self.dataset[f'{random_speaker}/{random_utt}/mel'][:].transpose(1,0)

        return {'speaker': self.speakers[index], 'tar': data_mel, 'tar_dmel':data_dmel, 'src': random_mel}


    def collate_fn(self, datas):
        batch = {}
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['len'] = [len(data['spectrogram']) for data in datas]
        batch['spectrogram'] = torch.tensor([self.pad_to_len(data['spectrogram'], max(batch['len'])) for data in datas])
        batch['len'] = [len(data['source']) for data in datas]
        batch['source'] = torch.tensor([self.pad_to_len(data['source'], max(batch['len'])) for data in datas])
        return batch

    def pad_to_len(self, arr, padded_len):
        padded_len = int(padded_len)
        padding = np.expand_dims(np.zeros_like(arr[0]),0)
        if (len(arr) < padded_len):
            for i in range(padded_len-len(arr)):
                arr = np.concatenate((arr,padding),axis=0)
        return arr[:padded_len]

    def __len__(self):
        return len(self.speakers)


class TotalDataset(Dataset):
    def __init__(self, h5_path, index_path, segment_size=128, load_mel=False, augment=False):
        self.dataset = None
        self.h5_path = h5_path
        self.augment = augment

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

        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')

        speaker_id, utt_id = self.total_indexes[index]
        
        data = {
            'speaker': speaker_id,  
            'mel': self.dataset[f'{speaker_id}/{utt_id}/mel'],
            'dmel': self.dataset[f'{speaker_id}/{utt_id}/dmel'],
            # N * 160 * 40
        }
        
        if self.segment_size is not None:
            t1 = random.randint(0, len(data['mel']) - self.segment_size)
            data['mel'] = data['mel'][t1:t1+self.segment_size]

            if self.augment:
                data['mel'] = spec_augment(data['mel'])

            if len(data['dmel'][t1:t1+self.segment_size]) < self.segment_size:
                if len(data['dmel']) > self.segment_size:
                    t2 = random.randint(0, len(data['dmel']) - self.segment_size)
                    data['dmel'] = data['dmel'][t2:t2+self.segment_size]
            else:
                data['dmel'] = data['dmel'][t1:t1+self.segment_size]
                
        return data

    def __len__(self):
        return len(self.total_indexes)

    def collate_fn(self, datas):
        batch = {}
        batch['speaker'] = [data['speaker'] for data in datas]
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


class TotalDataset_Avg(Dataset):
    def __init__(self, h5_path, index_path, segment_size=128, load_mel=False):
        self.dataset = None
        self.h5_path = h5_path

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

        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')

        speaker_id, utt_id1 = self.total_indexes[index]
        utt_id2 = utt_id1
        while utt_id2==utt_id1:
            utt_id2 = random.sample(self.class_indexes[speaker_id], 1)[0]
        
        data = {
            'speaker': speaker_id,  
            'mel1': self.dataset[f'{speaker_id}/{utt_id1}/mel'],
            'mel2': self.dataset[f'{speaker_id}/{utt_id2}/mel'],
            # N * 160 * 40
        }
        
        if self.segment_size is not None:
            t1 = random.randint(0, len(data['mel1']) - self.segment_size)
            t2 = random.randint(0, len(data['mel2']) - self.segment_size)
            data['mel1'] = data['mel1'][t1:t1+self.segment_size]
            data['mel2'] = data['mel2'][t2:t2+self.segment_size]

        return data

    def __len__(self):
        return len(self.total_indexes)

class TotalDataset_d(Dataset):
    def __init__(self, h5_path, index_path, segment_size=128, load_mel=False):
        self.dataset = None
        self.h5_path = h5_path

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
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')

        speaker_id, utt_id = self.total_indexes[index]
        
        data = {
            'speaker': speaker_id,  
            'lin': self.dataset[f'{speaker_id}/{utt_id}/lin'],
            'mel': self.dataset[f'{speaker_id}/{utt_id}/mel'],
            'dmel': self.dataset[f'{speaker_id}/{utt_id}/dmel'],
            # N * 160 * 40
        }
        
        if self.segment_size is not None:
            t1 = random.randint(0, len(data['lin']) - self.segment_size)
            data['mel'] = data['mel'][t1:t1+self.segment_size]

        mel_slices = self.dataset[f'{speaker_id}/{utt_id}/dmels'][:]
        data['dmel'] = np.array([data['dmel'][t0:t1] for t0, t1 in mel_slices])
                
        return data

    def __len__(self):
        return len(self.total_indexes)

    def collate_fn(self, datas):
        batch = {}
        batch['speaker'] = [data['speaker'] for data in datas]
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