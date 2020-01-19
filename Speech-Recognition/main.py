from scipy.io import wavfile
import pandas as pd
import numpy as np
from util import plot_classes_distrib, calculate_fft, plot_signals, \
    plot_fft, plot_fbank, plot_mfccs, envelope
from python_speech_features import mfcc, logfbank
import librosa


def main():
    print("Load data")
    # df "data frame"
    df = pd.read_csv('instruments.csv')
    df.set_index('fname', inplace=True)

    for f in df.index:
        rate, signal = wavfile.read('wavfiles/' + f)
        df.at[f, 'length'] = signal.shape[0] / rate

    classes = list(np.unique(df.length))
    df.reset_index(inplace=True)

    signals = {}
    fft = {}
    fbank = {}
    mfccs = {}

    for c in classes:
        wav_file = df[df.length == c].iloc[0, 0]
        signal, rate = librosa.load('wavfiles/' + wav_file, sr=44100)
        mask = envelope(signal, rate, .0005)
        signal = signal[mask]
        signals[c] = signal
        fft[c] = calculate_fft(signal, rate)
        bank = logfbank(signal[:rate], rate, nfilt=26, nfft=1103).T
        fbank[c] = bank
        mel = mfcc(signal[:rate], rate, numcep=13, nfilt=26, nfft=1103)
        mfccs[c] = mel

    plot_signals(signals)
    plot_fft(fft)
    plot_fbank(fbank)
    plot_mfccs(mfccs)


if __name__ == "__main__":
    main()
