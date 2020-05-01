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

        return data

    def __len__(self):
        return len(self.total_indexes)