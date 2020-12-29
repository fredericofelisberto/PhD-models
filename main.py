import argparse
from data import read_data
from pre_processing import group, features, build_features, clean
from plots import distribution_plt, signal_plt
from models import RNN, CONV
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import torch


def main():
    parse = argparse.ArgumentParser(description='Audio Classification Training')
    parse.add_argument('class_name', type=str, default='Gender',
                       help='The class name to train you model')
    parse.add_argument('model', choices=['conv', 'rnn'])
    parse.add_argument('features', choices=['stft', 'mfcc', 'cwt'])
    parse.add_argument('lng', choices=['cnh', 'eng'])
    parse.add_argument('-optimizer', choices=['sgd', 'adam'], default='adam')
    parse.add_argument('-cuda', action='store_true')
    # parse.add_argument('-clean', type=bool, default=False, help='To pre processing, remove low frequencies based on
    # threshold value')
    parse.add_argument('-dropout', type=int, default=0.5)
    parse.add_argument('-epochs', type=int, default=10, help='Number of epochs to train for')
    parse.add_argument('-threshold', type=int, default=0.005, help='Remove low frequencies below to')
    parse.add_argument('-nfilt', type=int, default=26, help='Number of filter')
    parse.add_argument('-nfeat', type=int, default=13)
    parse.add_argument('-nfft', type=int, default=1103)
    parse.add_argument('-rate', type=int, default=16000, help='Down sample rate to')
    parse.add_argument('-report', type=bool, default=False, help='Plot report')

    opt = parse.parse_args()
    # device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")
    # print(device)

    mod = None
    path_short = "cnh/"
    path_full = "data/cnh/"
    csv = "label_cnh.csv"

    if opt.lng == "eng":
        path_short = "eng/"
        path_full = "data/eng/"
        csv = "label_eng.csv"

    # if opt.clean:
    #     df_ = read_data()
    #     clean(df_, opt)

    df = read_data(path=path_short, csv=csv)
    x, y = build_features(df, opt=opt, path=path_full)
    input_shape = (x.shape[1], x.shape[2], 1) if opt.model == 'conv' else (x.shape[1], x.shape[2])
    y_flat = np.argmax(y, axis=1)
    weight = compute_class_weight('balanced', np.unique(y_flat), y_flat)
    class_weight = dict(enumerate(weight.flatten()))

    if opt.report:
        x_, x__, fft_, fbanks, mfccs, stf_f = features(df=df, opt=opt, path=path_full)
        distribution_plt(group(df, opt.class_name))
        signal_plt(x_, 'Time Series', 'tm')
        signal_plt(x__, 'Time Series ENVELOPE', 'tmenv')

    if opt.model == 'rnn':
        mod = RNN(input_shape, opt.optimizer)
    elif opt.model == 'conv':
        mod = CONV(input_shape, opt.optimizer)

    # if "cuda" in mod.device.type: torch.cuda.empty_cache()
    history = mod.fit(x, y, validation_split=0.50, batch_size=4, epochs=opt.epochs, class_weight=class_weight)

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss model')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/loss_loss.png')
    plt.show()

    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy model')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig('plots/acc_loss.png')
    plt.show()


if __name__ == "__main__":
    main()
