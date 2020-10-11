import numpy as np
import pandas as pd


def fft(signal, rate):
    y_length = len(signal)
    freq = np.fft.rfftfreq(y_length, d=1 / rate)
    Y = abs(np.fft.rfft(signal) / y_length)
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
