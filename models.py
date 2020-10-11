import numpy as np
from data import read


class RNN:
    def __init__(self, df,  n_filt=26, n_feat=13, n_fft=512, rate=16000, threshold=0.005):

        self.df = df
        self.nfilt = n_filt
        self.nfeat = n_feat
        self.nfft = n_fft
        self.rate = rate
        self.threshold = threshold
        self.classes = list(np.unique(df.Gender))
        self.class_dist = df.groupby(['Gender'])['length'].mean()
        self.n_samples = 2 * int(df['length'].sum() / 0.1)  # 10 seconds
        self.prob_dist = self.class_dist / self.class_dist.sum()
        self.choices = np.random.choice(self.class_dist.index, p=self.class_dist / self.class_dist.sum())
        self.step = int(rate / 10)
