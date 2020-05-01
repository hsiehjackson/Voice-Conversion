import librosa
import glob
import os
import re
import json
import h5py
import pysptk
import pyworld as pw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# mpl.use('Agg')
from collections import defaultdict
from params import Params
import librosa.display as lds
from audio import spectrogram2wav, wav2spectrogram
from math import ceil

root_dir='/home/mlpjb04/Voice_Conversion/Corpus-VCTK/'
data_dir='/home/mlpjb04/Voice_Conversion/h5py-VCTK/AutoVC_d'
audio_dir=os.path.join(root_dir,'wav48')
speaker2filenames = defaultdict(lambda : [])
for path in sorted(glob.glob(os.path.join(audio_dir, '*/*.wav'))):
    filename = path.strip().split('/')[-1]
    speaker_id, utt_id = re.match(r'p(\d+)_(\d+)\.wav', filename).groups()
    speaker2filenames[speaker_id].append(path)

# Split with different speaker
with open(os.path.join(data_dir, 'train_data.json')) as f:
    class_indexes = json.load(f)

# All data
total_indexes = []
for speaker_id, utt_ids in class_indexes.items():
    total_indexes += [[speaker_id,utt_id] for utt_id in utt_ids]

hparams = Params(os.path.join(data_dir, 'hp.json'))

def plot_spectrograms(data, pic_path):
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

def pad_seq(x, base=8):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant')


dataset = h5py.File(os.path.join(data_dir, 'dataset.hdf5'), 'r')


for sp, filelist in speaker2filenames.items():
    for filename in filelist:
        filename = '/home/mlpjb04/Voice_Conversion/hsieh_code_autoVC/hungyi/source/welcome_en3.wav'
        print(filename)
        mel, lin, mfcc, f0, out = wav2spectrogram(filename,hparams,display=True)
        # spectrogram2wav(mel, hparams, 'test.wav', use_mel=True)        
        # plt.subplot(3,1,1)
        # plt.plot(f0)
        # plt.subplot(3,1,2)
        # lds.specshow(mfcc.T, x_axis='time')
        # plt.subplot(3,1,3)
        # lds.specshow(mel.T, y_axis='mel', x_axis='time', cmap=plt.cm.Blues)
        # plt.show()
        input()




# for index in range(len(total_indexes)):
#     speaker_id, utt_id = total_indexes[index]
#     speaker_id = '229'
#     utt_id = '001'
#     print(speaker_id, utt_id)
#     mel_ori = dataset[f'{speaker_id}/{utt_id}/mel'][:]
#     plot_spectrograms(mel_ori,'test.png')
#     spectrogram2wav(mel_ori, hparams, 'test.wav', use_mel=True)

#     import torch.nn.functional as F
#     import torch

#     sample_low = [4,8,16,32]
#     sample_high = [64]
#     mel_pad = pad_seq(mel_ori, base=max(sample_high+sample_low))
#     mel_pad = torch.from_numpy(mel_pad)
#     mel_pad = mel_pad.unsqueeze(0)
#     mel_pad = mel_pad.permute(0,2,1)

#     mel_low = mel_pad[:,:30,:] 
#     mel_high = mel_pad[:,30:,:] 
    
#     mel_low_new = mel_low
#     mel_high_new = torch.zeros_like(mel_high)

#     # for s in sample_low:
#     #     mel_small = F.avg_pool1d(mel_low, kernel_size=s, ceil_mode=True)
#     #     mel_small = F.interpolate(mel_small, scale_factor=s, mode='nearest')
#     #     mel_low_new += mel_small

#     # for s in sample_high:
#     #     mel_small = F.avg_pool1d(mel_high, kernel_size=s, ceil_mode=True)
#     #     mel_small = F.interpolate(mel_small, scale_factor=s, mode='nearest')
#     #     mel_high_new += mel_small
    
#     # mel_high_new = mel_high_new / len(sample_high)
#     # mel_low_new = mel_low_new / len(sample_low)    

#     # mel_new = torch.cat((mel_high_new,mel_low_new),dim=1)
#     mel_new = torch.cat((mel_low_new,mel_high_new),dim=1)

#     mel_new = mel_new.permute(0,2,1)
#     mel_new = mel_new.numpy()
#     mel_new = mel_new.squeeze(0)
#     mel_new = mel_new[:mel_ori.shape[0],:]
#     plot_spectrograms(mel_new,'test_cnn.png')
#     spectrogram2wav(mel_new, hparams, 'test_cnn.wav', use_mel=True)
#     input()
