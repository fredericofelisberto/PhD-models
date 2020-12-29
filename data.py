import pandas as pd
from scipy.io import wavfile
from tqdm import tqdm


def read_data(root='data/', path='cnh/', csv='label_cnh.csv'):
    df = pd.read_csv(root + csv)
    df.set_index('Audio', inplace=True)
    for f in tqdm(df.index):
        rate, signal = wavfile.read(root + path + f)
        df.at[f, 'length'] = signal.shape[0] / rate
    return df
