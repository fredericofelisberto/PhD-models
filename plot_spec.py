import matplotlib.pyplot as plot
from scipy.io import wavfile
import pywt
import numpy as np


def main():
    num_steps = 128
    rate, signal = wavfile.read('data/phon/houstonw.wav')

    scales = np.arange(1, num_steps + 1)
    cwtmatr, freqs = pywt.cwt(signal[:, 1], scales, 'morl')

    plot.subplot(211)
    # plot.specgram(signal, rate)
    plot.specgram(signal[:, 1], rate)
    plot.xlabel('Time')
    plot.ylabel('Frequency')
    plot.savefig('plots/houstonw.png')
    plot.show()

    plot.subplot(212)
    plot.imshow(cwtmatr, extent=[-1, 1, 1, num_steps + 1], cmap='PRGn', aspect='auto',
                vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    plot.specgram(signal, rate)
    # plot.specgram(signal[:, 1], rate)
    # plot.xlabel('Time')
    # plot.ylabel('Frequency')
    plot.savefig('plots/cwt_houstonw.png')
    plot.show()


if __name__ == "__main__":
    main()
