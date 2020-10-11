import pandas as pd
from scipy.io import wavfile


def read(path='data'):
    df = pd.read_csv(path + '/label.csv', sep=';', engine='python')
    df.set_index(['Audio', 'AgeRange', 'Gender'], inplace=True)
    for f, a, g in df.index:
        rate, signal = wavfile.read(path + '/file/' + f)
        '''
        for g_ in ['Male', 'Female']:
            if g == g_ and a == 1:
                df.at[g, 'Both'] = 1
            elif g == g_ and a == 2:
                df.at[g, 'Both'] = 2
        '''
        df.at[f, 'length'] = signal.shape[0] / rate
    print(df)
    return df
