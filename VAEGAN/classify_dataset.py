import os
import json
import h5py
import torch
import random
import numpy as np
from torch.utils.data import Dataset

class ClassifyDataset(Dataset):
    def __init__(self, h5_path, index_path, segment_size, speaker2id=None):
        self.dataset = None
        self.h5_path = h5_path
        self.segment_size = segment_size

        # Split with different speaker
        with open(index_path) as f:
            self.class_indexes = json.load(f)
        # All data
        self.speaker2id={} 
        self.total_indexes = []
        for i, (speaker_id, utt_ids) in enumerate(self.class_indexes.items()):
            self.total_indexes += [[speaker_id,utt_id] for utt_id in utt_ids]
            self.speaker2id[speaker_id] = i

        # with open('/home/mlpjb04/Voice_Conversion/h5py-LibriTTS/AdaVAE_d/sp2id.json', 'w') as f:
        #     json.dump(self.speaker2id, f, indent=4)

        if speaker2id != None:
            self.speaker2id = speaker2id
        
        print('Total utterances: {}'.format(len(self.total_indexes)))

    def __getitem__(self, index):

        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')
            

        speaker_id, utt_id = self.total_indexes[index]
        label = int(self.speaker2id[speaker_id])
        
        data = {
            'speaker': speaker_id,  
            'mel': self.dataset[f'{speaker_id}/{utt_id}/mel'],
            'label': label
        }
        
        if self.segment_size is not None:
            t1 = random.randint(0, len(data['mel']) - self.segment_size)
            data['mel'] = data['mel'][t1:t1+self.segment_size]

        # print(data['speaker'])
        return data

    def __len__(self):
        return len(self.total_indexes)

    def collate_fn(self, datas):
        batch = {}
        batch['speaker'] = [data['speaker'] for data in datas]
        batch['mel'] = torch.tensor([data['mel'] for data in datas])
        batch['label'] = torch.tensor([data['label'] for data in datas])
        return batch


class ClassifyTestDateset(Dataset):
    def __init__(self, h5_path, speakers, utterances, indexes, speaker2id, segment_size):
        self.dataset = None
        self.h5_path = h5_path
        self.speakers = speakers
        self.utterances = utterances
        self.segment_size = segment_size
        self.indexes = indexes
        self.speaker2id = speaker2id

    def __getitem__(self,index):
        if self.dataset is None:
            self.dataset = h5py.File(self.h5_path, 'r')

        data_lin = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/lin'][:]
        data_mel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/mel'][:].transpose(1,0)
        data_dmel = self.dataset[f'{self.speakers[index]}/{self.utterances[index]}/dmel'][:]
        label = int(self.speaker2id[self.speakers[index]])

        random_speaker =  random.sample(set(list(self.indexes.keys())) - set(self.speakers[index]), 1)[0]
        random_utt = random.sample(self.indexes[random_speaker],1)[0]
        random_mel = self.dataset[f'{random_speaker}/{random_utt}/mel'][:].transpose(1,0)

        return {'speaker': self.speakers[index], 'tar': data_mel, 'tar_dmel':data_dmel, 'src': random_mel, 'label': label}

    def __len__(self):
        return len(self.speakers)