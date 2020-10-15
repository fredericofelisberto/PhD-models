import argparse
from data import read_data
from preProcessing import group, features, build_rand_feat, clean
from plots import distribution_plt, signal_plt
from models import RNN, fit


def main():
    parse = argparse.ArgumentParser(description='Audio Classification Training')
    parse.add_argument('class_name', type=str, default='Gender',
                       help='The class name to train you model')
    parse.add_argument('mode', choices=['conv', 'rnn'])

    parse.add_argument('--epochs', type=int, default=10)
    parse.add_argument('--threshold', type=int, default=0.005)
    parse.add_argument('--nfilt', type=int, default=26)
    parse.add_argument('--nfeat', type=int, default=13)
    parse.add_argument('--nfft', type=int, default=1103)
    parse.add_argument('--rate', type=int, default=16000)

    opt = parse.parse_args()
    df = read_data()
    # df.reset_index(level='Audio', inplace=True)
    clean(df, opt)

    X, y = build_rand_feat(df, opt=opt)

    input_shape = (X.shape[1], X.shape[2], 1) if opt.mode == 'conv' else (X.shape[1], X.shape[2])
    model = RNN(input_shape)

    fit(model, X, y, opt.epochs)

    '''
    X_, X_env, fft_, fbanks, mfccs = features(df, opt)

    # plot class distribution
    distribution(group(df, opt.class_name))
    signal(X_, 'Time Series')
    signal(X_env, 'Time Series cleaned')
    '''


if __name__ == "__main__":
    main()
