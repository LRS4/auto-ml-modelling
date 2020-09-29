import pytest
import os
import pandas as pd
from model import load_dataset, build_model
from transformers import TrainTestSplitter, EmbarkedAdjuster

from config import (X_TRAIN, Y_TRAIN, DATA_DIR,
                    DATA_FILENAME, TARGET_COLUMN_NAME, PARAMS)

def test_build_model():
    # arrange
    X_train, y_train = load_dataset(X_TRAIN, Y_TRAIN)
    model = build_model()
    print(model)
    # act
    model.fit(X_train, y_train)
    #model.predict(X_train)

    # assert
    

def test_embarked_adjuster():
    # arrange
    X_train, y_train = load_dataset(X_TRAIN, Y_TRAIN)
    ea = EmbarkedAdjuster()

    # act
    ea.fit(X_train)
    df = ea.transform(X_train)

    # assert
    for column in df.columns:
        assert column in ['Embarked']
    assert type(y_train) == pd.DataFrame


# def test_accommodation_extractor():
#     # arrange
#     X_train, y_train = load_dataset(X_TRAIN, Y_TRAIN)
#     ae = AcommodationExtractor()

#     # act
#     ae.fit(X_train)
#     df = ae.transform(X_train)

#     # assert
#     for column in df.columns:
#         assert column in ['hasCabin', 'FamilySize', 'Title']
#     assert type(y_train) == pd.DataFrame


def test_train_test_splitter():
    # arrange
    tts = TrainTestSplitter()

    # act
    splits = tts.split_dataset(save_splits=False, print_summary=True)
    complete_dataset = pd.read_csv(
        os.sep.join([DATA_DIR, DATA_FILENAME])).shape[0]
    train_proportion = (splits['X_train'].shape[0] / complete_dataset)

    # assert
    assert train_proportion + PARAMS['test_size'] >= 0.999
