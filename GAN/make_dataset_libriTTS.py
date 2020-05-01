import warnings
warnings.filterwarnings("ignore")
import json
import h5py
import numpy as np
import sys
import os
import glob
import random
import re
from collections import defaultdict
from tqdm import tqdm
from multiprocessing import Pool
import itertools

from tacotron.utils import get_spectrograms
from resemblyzer import preprocess_wav
from resemblyzer import audio
from resemblyzer.hparams import *
from pathlib import Path

TEST_SPEAKERS = 20
DEV_UTTERANCE_PROPORTION = 0.1
SEGMENT_SIZE = 128
WORKERS = 30
finish = 0

def read_speaker_info(speaker_info_path):
    speaker_infos = {}
    man, woman = [], []
    with open(speaker_info_path, 'r') as f:
        head = f.readline().strip().split()
        for i, line in enumerate(f):
            speaker_id = line.strip().split()[0]
            speaker_info = line.strip().split()[1:]
            speaker_infos[speaker_id] = speaker_info
            if speaker_info[1] == 'F':
                woman.append(speaker_id)
            elif speaker_info[1] == 'M':
                man.append(speaker_id)
    return speaker_infos, head, man, woman

def read_filenames(root_dir):
    speaker2filenames = defaultdict(lambda : [])
    for path in tqdm(sorted(glob.glob(os.path.join(root_dir, '*/*/*.wav')))):
        filename = path.strip().split('/')[-1]
        speaker_id = re.match(r'(\d+)_(\d+)_(\d+)_(\d+)\.wav', filename).groups()[0]
        speaker2filenames[speaker_id].append(path)
    return speaker2filenames


def compute_partial_slices(n_samples, rate=1.3, min_coverage=0.75):
    assert 0 < min_coverage <= 1
    
    # Compute how many frames separate two partial utterances
    samples_per_frame = int((sampling_rate * mel_window_step / 1000))
    n_frames = int(np.ceil((n_samples + 1) / samples_per_frame))
    frame_step = int(np.round((sampling_rate / rate) / samples_per_frame))
    assert 0 < frame_step, "The rate is too high"
    assert frame_step <= partials_n_frames, "The rate is too low, it should be %f at least" % \
        (sampling_rate / (samples_per_frame * partials_n_frames))
    
    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - partials_n_frames + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + partials_n_frames])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))
    
    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (n_samples - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]
    
    return wav_slices, mel_slices

def d_wav2spec(wav):
    wav_slices, mel_slices = compute_partial_slices(len(wav))

    max_wave_length = wav_slices[-1].stop
    if max_wave_length >= len(wav):
        wav = np.pad(wav, (0, max_wave_length - len(wav)), "constant")
    
    # Split the utterance into partials and forward them through the model
    mel = audio.wav_to_mel_spectrogram(wav)
    mels = np.array([mel[s] for s in mel_slices])

    return mel

def universal_worker(input_pair):
    function, args = input_pair
    return function(*args)

def pool_args(function, *args):
    return zip(itertools.repeat(function), zip(*args))

def make_one_dataset(filename,total,display=False):
    global finish
    sub_filename = filename.strip().split('/')[-1]
    groups = re.match(r'(\d+)_(\d+)_(\d+)_(\d+)\.wav', sub_filename).groups()     # format: p{speaker}_{sid}.wav
    speaker_id = groups[0]
    utt_id = '_'.join(groups[1:])
    mel_spec, lin_spec = get_spectrograms(filename)

    wav = preprocess_wav(Path(filename))
    d_mel = d_wav2spec(wav)

    print('[Processor] - processing {}/{} s{}-{} | mel: {} | d_mel: {}'.format(
       finish*WORKERS, total, speaker_id, utt_id, mel_spec.shape, d_mel.shape), end='\r')
    result = {}
    result['speaker_id'] = speaker_id
    result['utt_id'] = utt_id
    result['d_mel_spec'] = d_mel
    result['mel_spec'] = mel_spec
    result['lin_spec'] = lin_spec
    finish += 1
    return result

if __name__ == '__main__':
    random.seed(42)

    if len(sys.argv) < 2:
        print('Usage: python3 make_dataset_vctk.py [folder_name]')
        exit(0)

    folder_name=sys.argv[1]
    root_dir='/home/mlpjb04/Voice_Conversion/Corpus-LibriTTS/'
    save_dir=os.path.join('/home/mlpjb04/Voice_Conversion/h5py-LibriTTS/',folder_name)
    os.makedirs(save_dir,exist_ok=True)

    train_audio_dir=os.path.join(root_dir,'train-clean-100')
    test_audio_dir=os.path.join(root_dir,'dev-clean')
    h5py_path=os.path.join(save_dir,'dataset.hdf5')


    train_speaker2filenames = read_filenames(train_audio_dir)
    test_speaker2filenames = read_filenames(test_audio_dir)

    # Prepare Split List
    train_path_list, in_test_path_list, out_test_path_list = [], [], []

    for speaker, path_list in tqdm(train_speaker2filenames.items()):
        random.shuffle(path_list)
        dev_data_size = int(len(path_list) * DEV_UTTERANCE_PROPORTION)
        train_path_list += path_list[:-dev_data_size]
        in_test_path_list += path_list[-dev_data_size:]

    for speaker, path_list in tqdm(test_speaker2filenames.items()):
        out_test_path_list += path_list
    

    with h5py.File(h5py_path, 'a') as f_h5:

        for datatype, data_list in zip(['train', 'in_test', 'out_test'], [train_path_list, in_test_path_list, out_test_path_list]):

            if datatype == 'train':
                continue

            P = Pool(processes=WORKERS) 
            results = P.map(universal_worker, pool_args(make_one_dataset, 
                                                    data_list, 
                                                    [len(data_list)]*len(data_list)
                                                    ))
            P.close()
            P.join()

            total_segment = 0
            savenames = {}
            for i in tqdm(range(len(results))):
                if len(results[i]['mel_spec']) >= SEGMENT_SIZE:
                    speaker_id = results[i]['speaker_id']
                    utt_id = results[i]['utt_id']
                    f_h5.create_dataset(f'{speaker_id}/{utt_id}/dmel', data=results[i]['d_mel_spec'], dtype=np.float32)
                    f_h5.create_dataset(f'{speaker_id}/{utt_id}/mel', data=results[i]['mel_spec'], dtype=np.float32)
                    f_h5.create_dataset(f'{speaker_id}/{utt_id}/lin', data=results[i]['lin_spec'], dtype=np.float32)
                    # f_h5.create_dataset(f'{speaker_id}/{utt_id}/aud', data=results[i]['audio'], dtype=np.float32)
                    if speaker_id in savenames:
                        savenames[speaker_id].append(utt_id)
                    else:
                        savenames[speaker_id] = [utt_id]
                    total_segment += 1

            for k in list(savenames.keys()):
                savenames[k].sort()

            with open(os.path.join(save_dir, '{}_data.json'.format(datatype)), 'w') as f:
                json.dump(savenames,f,indent=4)

            print('{} sets have {} segments'.format(datatype,total_segment))

    # all_path_list = train_path_list + in_test_path_list + out_test_path_list

    # with h5py.File(h5py_path, 'w') as f_h5:

    #     P = Pool(processes=WORKERS) 
    #     results = P.map(universal_worker, pool_args(make_one_dataset, 
    #                                             all_path_list, 
    #                                             [len(all_path_list)]*len(all_path_list)
    #                                             ))
    #     P.close()
    #     P.join()

    #     train_path_result = results[:len(train_path_list)]
    #     in_test_path_result = results[len(train_path_list):len(train_path_list)+len(in_test_path_list)]
    #     out_test_path_result = results[-len(out_test_path_list):]

    #     for datatype, results in zip(['train', 'in_test', 'out_test'], [train_path_result, in_test_path_result, out_test_path_result]):
    #         total_segment = 0
    #         savenames = {}
    #         for i in tqdm(range(len(results))):
    #             if len(results[i]['mel_spec']) >= SEGMENT_SIZE:
    #                 speaker_id = results[i]['speaker_id']
    #                 utt_id = results[i]['utt_id']
    #                 f_h5.create_dataset(f'{speaker_id}/{utt_id}/dmel', data=results[i]['d_mel_spec'], dtype=np.float32)
    #                 f_h5.create_dataset(f'{speaker_id}/{utt_id}/mel', data=results[i]['mel_spec'], dtype=np.float32)
    #                 f_h5.create_dataset(f'{speaker_id}/{utt_id}/lin', data=results[i]['lin_spec'], dtype=np.float32)
    #                 # f_h5.create_dataset(f'{speaker_id}/{utt_id}/aud', data=results[i]['audio'], dtype=np.float32)
    #                 if speaker_id in savenames:
    #                     savenames[speaker_id].append(utt_id)
    #                 else:
    #                     savenames[speaker_id] = [utt_id]
    #                 total_segment += 1

    #         for k in list(savenames.keys()):
    #             savenames[k].sort()

    #         with open(os.path.join(save_dir, '{}_data.json'.format(datatype)), 'w') as f:
    #             json.dump(savenames,f,indent=4)

    #         print('{} sets have {} segments'.format(datatype,total_segment))