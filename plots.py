import matplotlib.pyplot as plt


def signal_plt(signal_, title):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
    fig.suptitle(title, size=16)
    for x in range(2):
        ax[x].set_title(list(signal_.keys())[x])
        ax[x].plot(list(signal_.values())[x])
        ax[x].get_xaxis().set_visible(False)
        ax[x].get_yaxis().set_visible(False)
    # plt.savefig('plots/signals.png')
    plt.show()


def fft_plt(fft_):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20,5))
    fig.suptitle('fft', size=16)
    for x in range(2):
        data = list(fft_.values())[x]
        Y, freq = data[0], data[1]
        ax[x].set_title(list(fft_.keys())[x])
        ax[x].plot(freq, Y)
        ax[x].get_xaxis().set_visible(False)
        ax[x].get_yaxis().set_visible(False)
    plt.show()


def mFccs_fBanks_plt(data, title, name):
    fig, axes = plt.subplots(ncols=3, figsize=(20, 5))
    fig.suptitle(title, size=16)
    i = 0
    for y in range(3):
        axes[y].set_title(list(data.keys())[i])
        axes[y].imshow(list(data.values())[i],
                       cmap='hot', interpolation='nearest')
        axes[y].get_xaxis().set_visible(False)
        axes[y].get_yaxis().set_visible(False)
        i += 1
    # plt.savefig('plots/' + name + '.png')
    plt.show()


def distribution_plt(cd):
    print(cd)
    fig, x_axis = plt.subplots()
    x_axis.set_title('Class distribution', y=1)
    x_axis.pie(cd, labels=cd.index, autopct='%1.1f%%', shadow=True, startangle=90)
    x_axis.axis('equal')
    plt.show()
    # plt.savefig('plots/class_distr.pdf')
