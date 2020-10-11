import matplotlib.pyplot as plt


def signal(signal_):
    fig, axes = plt.subplots(ncols=3, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Time Series', size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(signal_.keys())[i])
        axes[y].plot(list(signal_.values())[i])
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1
    plt.savefig('plots/signals.png')


def fft(fft_):
    fig, axes = plt.subplots(ncols=3, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle('Fourier Transforms', size=16)
    i = 0
    for y in range(3):
        data = list(fft_.values())[i]
        Y, freq = data[0], data[1]
        axes[y].set_title(list(fft_.keys())[i])
        axes[y].plot(freq, Y)
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1
    plt.savefig('plots/fft.png')


def mFccs_fBanks(data, title, name):
    fig, axes = plt.subplots(ncols=3, sharex=False,
                             sharey=True, figsize=(20, 5))
    fig.suptitle(title, size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(data.keys())[i])
        axes[y].imshow(list(data.values())[i],
                       cmap='hot', interpolation='nearest')
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1
    plt.savefig('plots/' + name + '.png')


def distribution(cd):
    print(cd)
    fig, x_axis = plt.subplots()
    x_axis.set_title('Class distribution', y=1)
    x_axis.pie(cd, labels=cd.index, autopct='%1.1f%%', shadow=True, startangle=90)
    x_axis.axis('equal')
    plt.show()
    # plt.savefig('plots/class_distr.pdf')
