import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_signals(signals):
    fig, axes = plt.subplots(ncols=3, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(signals.keys())[i])
        axes[y].plot(list(signals.values())[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1
    plt.savefig('plots/signals.png')


def plot_fft(fft):
    fig, axes = plt.subplots(ncols=3, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for y in range(3):
        data = list(fft.values())[i]
        Y, freq = data[0], data[1]
        axes[y].set_title(list(fft.keys())[i])
        axes[y].plot(freq, Y)
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1
    plt.savefig('plots/fft.png')


def plot_fbank(fbank):
    fig, axes = plt.subplots(ncols=3, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Filter Bank Coefficients', size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(fbank.keys())[i])
        axes[y].imshow(list(fbank.values())[i],
                          cmap='hot', interpolation='nearest')
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1
    plt.savefig('plots/fbank.png')


def plot_mfccs(mfccs):
    fig, axes = plt.subplots(ncols=3, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Mel Frequency Cepstrum Coefficients', size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(mfccs.keys())[i])
        axes[y].imshow(list(mfccs.values())[i],
                          cmap='hot', interpolation='nearest')
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1
    plt.savefig('plots/mfccs.png')


def calculate_fft(y, rate):
    y_length = len(y)
    freq = np.fft.rfftfreq(y_length, d=1 / rate)
    Y = abs(np.fft.rfft(y) / y_length)
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


def plot_classes_distrib(data):
    fig, x_axis = plt.subplots()
    x_axis.set_title('Class distribuition', y=1)
    x_axis.pie(data, labels=data.index, autopct='%1.1f%%', shadow=True, startangle=90)
    x_axis.axis('equal')
    plt.savefig('plots/class_distr.pdf')
