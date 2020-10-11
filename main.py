import argparse
from data import read
from processing import group
from plots import distribution


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('class_name', choices=['Gender', 'AgeRange', 'Both'],
                       help='The class name to train you model')

    opt = parse.parse_args()

    df = read()

    # plot class distribution
    distribution(group(df, opt.class_name))


if __name__ == "__main__":
    main()
