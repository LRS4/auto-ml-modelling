import argparse

from transformers import TrainTestSplitter


def main():
    parser = argparse.ArgumentParser(
        description="A command line tool to manage the model")
    parser.add_argument('stage',
                        metavar='stage',
                        type=str,
                        choices=['split', 'train', 'test'],
                        help="Stage to run.")

    stage = parser.parse_args().stage

    if stage == 'split':
        train_test_splitter = TrainTestSplitter()
        train_test_splitter.split_dataset()

    if stage == 'train':
        print("Training model...")

    elif stage == "test":
        print("Testing model...")


if __name__ == "__main__":
    main()