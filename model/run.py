import argparse

from transformers import TrainTestSplitter
from model import train_model, test_model, predict


def main():
    parser = argparse.ArgumentParser(
        description="A command line tool to manage the model")
    parser.add_argument('stage',
                        metavar='stage',
                        type=str,
                        choices=['split', 'train', 'test', 'predict'],
                        help="Stage to run.")

    stage = parser.parse_args().stage

    if stage == 'split':
        train_test_splitter = TrainTestSplitter()
        train_test_splitter.split_dataset()

    if stage == 'train':
        print('Training model...', end="\n\n")
        train_model()

    if stage == 'test':
        print('Testing model...', end="\n\n")
        test_model()

    if stage == 'predict':
        print('Outputting predictions to ../outputs/predictions.csv', end="\n\n")
        predict()

if __name__ == "__main__":
    main()