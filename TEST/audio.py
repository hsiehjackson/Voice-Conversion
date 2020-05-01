import librosa
import librosa.filters
import numpy as np
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display as lds
import pysptk
from sklearn.preprocessing import scale
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def plot_spectrogram(M, L, mfcc, preporcessed, f0):
    # Linear
    plt.subplot(2, 2, 1)
    # lds.specshow(L.T, y_axis='linear', x_axis='time', cmap=plt.cm.Blues)
    # plt.colorbar(format='%+2.0f dB')
    plt.plot(f0)
    plt.title('f0 estimation')
    # Mel
    plt.subplot(2, 2, 2)
    lds.specshow(M.T, y_axis='mel', x_axis='time', cmap=plt.cm.Blues)
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    # Original Audio
    plt.subplot(2, 2, 3)
    lds.specshow(mfcc.T, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    # Preporcessed Audio
    plt.subplot(2, 2, 4)
    plt.plot(preporcessed)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    plt.title('Preporcessed Audio')
    plt.show()


def wav2numpy(inpath,hparams):
    wav = load_wav(inpath, sr=hparams.sample_rate)
    if hparams.trim_silence:
        wav = trim_silence(wav, hparams)
    return wav

def wav2wav(inpath,hparams,outpath):
    wav = load_wav(inpath, sr=hparams.sample_rate)
    if hparams.trim_silence:
        wav, index = trim_silence(wav, hparams)
    save_wav(wav,outpath, sr=hparams.sample_rate)


def spectrogram2wav(spectrogram, hparams, fpath ,use_mel=True):
    spectrogram = spectrogram.T
    if use_mel:
        wav = inv_mel_spectrogram(spectrogram,hparams)
    else:
        wav = inv_linear_spectrogram(spectrogram,hparams)

    if hparams.trim_silence:
        wav, index = trim_silence(wav, hparams)

    save_wav(wav, fpath, sr=hparams.sample_rate)


def wav2spectrogram(fpath,hparams,display=False, silence=None):
    # Loading sound file
    wav = load_wav(fpath, sr=hparams.sample_rate)
    if hparams.trim_silence:
        wav, index = trim_silence(wav, hparams, silence)
    preem_wav = preemphasis(wav, hparams)

    # Rescale wav
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
        preem_wav = preem_wav / np.abs(preem_wav).max() * hparams.rescaling_max

        #Assert all audio is in [-1, 1]
        if (wav > 1.).any() or (wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))
        if (preem_wav > 1.).any() or (preem_wav < -1.).any():
            raise RuntimeError('wav has invalid value: {}'.format(wav_path))


    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = melspectrogram(preem_wav, hparams).astype(np.float32).T # (T, n_mels)
    mel_frames = mel_spectrogram.shape[0]

    # Compute the linear scale spectrogram from the wav
    linear_spectrogram = linearspectrogram(preem_wav, hparams).astype(np.float32).T # (T, 1+n_fft//2)
    linear_frames = linear_spectrogram.shape[0]

    mfcc = mfccgram(mel_spectrogram, hparams)
    mfcc_frames = mfcc.shape[0]

    f0 = f0gram(fpath,hparams,index)
    f0_frames = f0.shape[0]

    # print(mel_spectrogram.shape, linear_spectrogram.shape, mfcc.shape, f0.shape)

    if linear_frames != f0_frames:
        len_pad = linear_frames - f0_frames
        f0 = np.pad(f0, (0, len_pad), 'constant', constant_values=(0))
        f0_frames = f0.shape[0]
        # if len_pad > 1:
            # print(fpath)

    # assert linear_frames == mel_frames
    # assert linear_frames == mfcc_frames
    # assert linear_frames == f0_frames

    constant_values = 0.
    out = wav
    out_dtype = np.float32

    # For audio
    if hparams.use_lws:
        #Ensure time resolution adjustement between audio and mel-spectrogram
        fft_size = hparams.n_fft if hparams.win_size is None else hparams.win_size
        l, r = lws_pad_lr(wav, hparams.n_fft, get_hop_size(hparams))
        #Zero pad audio signal
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)
    else:
        #Ensure time resolution adjustement between audio and mel-spectrogram
        l, r = librosa_pad_lr(wav, hparams.n_fft, get_hop_size(hparams))
        #Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
        out = np.pad(out, (l, r), mode='constant', constant_values=constant_values)

    out = out.astype(out_dtype)
    assert len(out) >= mel_frames * get_hop_size(hparams)

    #time resolution adjustement
    #ensure length of raw audio is multiple of hop size so that we can use
    #transposed convolution to upsample
    out = out[:mel_frames * get_hop_size(hparams)]
    assert len(out) % get_hop_size(hparams) == 0

    if display:
        plot_spectrogram(mel_spectrogram, linear_spectrogram, mfcc, out, f0)

    return mel_spectrogram, linear_spectrogram, mfcc, f0, out


# Load and Save .wav
def load_wav(path, sr):
    return librosa.core.load(path, sr=sr)[0]

def save_wav(wav, path, sr):
    wav *= 32767 / max(0.01, np.max(np.abs(wav)))
    wavfile.write(path, sr, wav.astype(np.int16))

# f0 estimation()
def f0gram(filename,hparams,index):
    fs, x = wavfile.read(filename)
    if x.ndim > 1:
        x = np.mean(x,axis=1) 

    x = librosa.core.resample(x.astype(np.float32),fs,hparams.sample_rate,'kaiser_best')
    x = x[index[0]:index[1]]
    f0_rapt = pysptk.rapt(x, fs=hparams.sample_rate, hopsize=256, min=10, max=7600, otype="f0")
    return f0_rapt

# mfcc
def mfccgram(mel_spec,hparams):
    mfcc_delta0 = librosa.feature.mfcc(S=(mel_spec.T), dct_type=hparams.dct_type, n_mfcc=hparams.n_mfcc)
    mfcc_delta1 = librosa.feature.delta(mfcc_delta0, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc_delta0, order=2)
    mfcc = np.concatenate((mfcc_delta0,mfcc_delta1,mfcc_delta2),axis=0)
    # mfcc = scale(mfcc,axis=1)
    return mfcc.T

# spectrogram
def linearspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = _amp_to_db(np.abs(D)**hparams.magnitude_power, hparams) - hparams.ref_level_db
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def melspectrogram(wav, hparams):
    D = _stft(wav, hparams)
    S = _amp_to_db(_linear_to_mel(np.abs(D)**hparams.magnitude_power, hparams), hparams) - hparams.ref_level_db
    if hparams.signal_normalization:
        return _normalize(S, hparams)
    return S

def inv_linear_spectrogram(linear_spectrogram, hparams):
    '''Converts linear spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(linear_spectrogram, hparams)
    else:
        D = linear_spectrogram

    S = _db_to_amp(D + hparams.ref_level_db)**(1/hparams.magnitude_power) #Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
    else:
        y = _griffin_lim(S ** hparams.power, hparams)

    if hparams.preemphasis:
        return inv_preemphasis(y, hparams)
    else:
        return y


def inv_mel_spectrogram(mel_spectrogram, hparams):
    '''Converts mel spectrogram to waveform using librosa'''
    if hparams.signal_normalization:
        D = _denormalize(mel_spectrogram, hparams)
    else:
        D = mel_spectrogram

    S = _mel_to_linear(_db_to_amp(D + hparams.ref_level_db)**(1/hparams.magnitude_power), hparams)  # Convert back to linear

    if hparams.use_lws:
        processor = _lws_processor(hparams)
        D = processor.run_lws(S.astype(np.float64).T ** hparams.power)
        y = processor.istft(D).astype(np.float32)
    else:
        y = _griffin_lim(S ** hparams.power, hparams)

    if hparams.preemphasis:
        return inv_preemphasis(y, hparams)
    else:
        return y

# griffin_lim
def _griffin_lim(S, hparams):
    '''librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    '''
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = _istft(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(_stft(y, hparams)))
        y = _istft(S_complex * angles, hparams)
    return y

# trim Silence 
def trim_silence(wav, hparams, silence=None):
    top_db = silence if silence != None else 60
    wav, index = librosa.effects.trim(wav,top_db=top_db)
    return wav, index
    
# preemphasis
def preemphasis(wav, hparams):
    if hparams.preemphasis is not None:
        return signal.lfilter([1, -hparams.preemphasis], [1], wav)
    return wav

def inv_preemphasis(wav, hparams):
    if hparams.preemphasis is not None:
        return signal.lfilter([1], [1, -hparams.preemphasis], wav)
    return wav

def _stft(y, hparams):
    if hparams.use_lws:
        return _lws_processor(hparams).stft(y).T
    else:
        return librosa.stft(y=y, n_fft=hparams.n_fft, hop_length=get_hop_size(hparams), win_length=hparams.win_size, pad_mode='constant')

def _istft(y, hparams):
    return librosa.istft(y, hop_length=get_hop_size(hparams), win_length=hparams.win_size)

def get_hop_size(hparams):
    hop_size = hparams.hop_size
    if hop_size is None:
        assert hparams.frame_shift_ms is not None
        hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    return hop_size


######################################## librosa package
def librosa_pad_lr(x, fsize, fshift, pad_sides=1):
    '''compute right padding (final frame) or both sides padding (first and final frames)
    '''
    assert pad_sides in (1, 2)
    # return int(fsize // 2)
    pad = (x.shape[0] // fshift + 1) * fshift - x.shape[0]
    if pad_sides == 1:
        return 0, pad
    else:
        return pad // 2, pad // 2 + pad % 2

######################################## lws package

def _lws_processor(hparams):
    import lws
    return lws.lws(hparams.n_fft, get_hop_size(hparams), mode="speech")


def lws_num_frames(length, fsize, fshift):
    """Compute number of time frames of lws spectrogram
    """
    pad = (fsize - fshift)
    if length % fshift == 0:
        M = (length + pad * 2 - fsize) // fshift + 1
    else:
        M = (length + pad * 2 - fsize) // fshift + 2
    return M


def lws_pad_lr(x, fsize, fshift):
    """Compute left and right padding lws internally uses
    """
    M = lws_num_frames(len(x), fsize, fshift)
    pad = (fsize - fshift)
    T = len(x) + 2 * pad
    r = (M - 1) * fshift + fsize - T
    return pad, pad + r


######################################## Conversions
_mel_basis = None
_inv_mel_basis = None

def _linear_to_mel(spectrogram,hparams):
    global _mel_basis
    if _mel_basis is None:
        _mel_basis = _build_mel_basis(hparams)
    return np.dot(_mel_basis, spectrogram)

def _mel_to_linear(mel_spectrogram, hparams):
    global _inv_mel_basis
    if _inv_mel_basis is None:
        _inv_mel_basis = np.linalg.pinv(_build_mel_basis(hparams))
    return np.maximum(1e-10, np.dot(_inv_mel_basis, mel_spectrogram))

def _build_mel_basis(hparams):
    assert hparams.fmax <= hparams.sample_rate // 2
    return librosa.filters.mel(hparams.sample_rate, hparams.n_fft,
                               fmin=hparams.fmin, fmax=hparams.fmax,
                               n_mels=hparams.num_mels)


def _amp_to_db(x,hparams):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S,hparams):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S,hparams):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db