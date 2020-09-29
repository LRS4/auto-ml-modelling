import argparse

from transformers import TrainTestSplitter


def main():
    parser = argparse.ArgumentParser(
        description="A command line tool to manage the project")
    parser.add_argument('stage',
                        metavar='stage',
                        type=str,
                        choices=['tune', 'train', 'test', 'split'],
                        help="Stage to run.")

    stage = parser.parse_args().stage

    if stage == 'split':
        tts = TrainTestSplitter()
        tts.split_dataset()

    if stage == 'tune':
        print("Tuning for optimal pipeline...")

    elif stage == "test":
        print("test argument hit!")


if __name__ == "__main__":
    main()