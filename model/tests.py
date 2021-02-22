import pytest
import os
import pandas as pd

from model import build_model
from transformers import TrainTestSplitter, TrainDataPreprocessor
from transformers import TestDataPreprocessor as testDataPreprocessor


def test_build_model():
    # arrange
    model = build_model()

    # act
    model_as_text = str(model)

    # assert
    assert 'RandomForestClassifier' in model_as_text
    

def test_train_model():
    pass


def test_test_model():
    pass


def test_predict():
    pass


def test_train_test_splitter():
    pass


def test_train_data_preprocessor():
    pass


def test_test_data_preprocessor():
    # arrange
    test_data_preprocessor = testDataPreprocessor()

    # act 
    X_test, y_test = test_data_preprocessor.get_prepared_test_data()

    # assert
    assert len(X_test) > 150