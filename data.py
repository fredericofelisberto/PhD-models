import pandas as pd
from scipy.io import wavfile


def read_data(path='data'):
    df = pd.read_csv(path + '/label.csv')
    df.set_index('Audio', inplace=True)
    for f in df.index:
        rate, signal = wavfile.read(path + '/file/' + f)
        df.at[f, 'length'] = signal.shape[0] / rate
    return df
