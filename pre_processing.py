import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from scipy.io import wavfile
import librosa
from python_speech_features import mfcc, logfbank
from tensorflow.keras.utils import to_categorical
from scipy import signal
import pywt


def fft(sig, rate):
    y_length = len(sig)
    freq = np.fft.rfftfreq(y_length, d=1 / rate)
    Y = abs(np.fft.rfft(sig) / y_length)
    return Y, freq


def envelope(y, rate, threshold):
    mask = []
    y = pd.Series(y).apply(np.abs)
    y_mean = y.rolling(window=int(rate / 10), min_periods=1, center=True).mean()
    for mean in y_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return mask


def group(df, class_name):
    if class_name == 'Both':
        # TODO for both classes selected
        return ''
    else:
        return df.groupby([class_name])["length"].mean()


def clean(df, opt, path='data/'):
    if len(os.listdir(path + 'fileClean')) == 0:
        df.reset_index(inplace=True)
        for f in tqdm(df.Audio):
            s, r = librosa.load(path + 'file/' + f, sr=opt.rate)
            mask = envelope(s, r, opt.threshold)
            wavfile.write(filename=path + 'fileClean/' + f, rate=r, data=s[mask])


def features(df, opt, path):
    df.reset_index(inplace=True)
    sign = {}
    signal_env = {}
    ffts = {}
    fbanks = {}
    mfccs = {}
    stf_f = {}

    for c in list(np.unique(df.Gender)):
        wav = df[df.Gender == c].iloc[0, 0]
        s, r = librosa.load(path + wav)
        sign[c] = s

        mask = envelope(s, r, opt.threshold)
        s = s[mask]
        signal_env[c] = s
        ffts[c] = fft(s, r)

        _, _, Zxx = signal.stft(s, fs=r, window='hann', nperseg=256)
        stf_f[c] = Zxx

        fbanks[c] = logfbank(s[:r], r, nfilt=opt.nfilt, nfft=1103).T

        mfccs[c] = mfcc(s[:r], r, numcep=opt.nfeat, nfilt=opt.nfilt, nfft=opt.nfft).T

    return sign, signal_env, fft, fbanks, mfccs, stf_f


def build_features(df, opt, path):
    X = []
    y = []
    num_steps = 128
    scales = np.arange(1, num_steps + 1)

    class_dist = group(df, 'Gender')
    prob_disturb = class_dist / class_dist.sum()
    n_samples = 2 * int(df['length'].sum() / 0.1)

    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rd_class = np.random.choice(class_dist.index, p=prob_disturb)
        file = np.random.choice(df[df.Gender == rd_class].index)
        r, wav = wavfile.read(path + file)
        label = df.at[file, 'Gender']
        rand_index = np.random.randint(0, wav.shape[0] - int(opt.rate / 10))
        sample = wav[rand_index:rand_index + int(opt.rate / 10)]

        X_sample = None
        if opt.features == 'mfcc':
            X_sample = mfcc(sample, opt.rate, numcep=opt.nfeat, nfilt=opt.nfilt, nfft=opt.nfft)

        elif opt.features == 'stft':
            _, _, Zxx = signal.stft(sample, fs=r, window='hann', nperseg=256)
            X_sample = np.abs(Zxx)

        elif opt.features == 'cwt':
            coef, freqs = pywt.cwt(sample, scales, 'morl', method='conv')
            X_sample = coef

        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if opt.model == 'conv' else X_sample.T)
        y.append(list(np.unique(df.Gender)).index(label))
    X, y = np.array(X), np.array(y)
    X = (X - _min) / (_max - _min)
    if opt.model == 'conv':
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
    else:
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2])
    y = to_categorical(y, num_classes=2)
    return X, y
