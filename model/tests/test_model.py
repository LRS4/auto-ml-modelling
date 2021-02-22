import pytest
import os
import pandas as pd

from model.model import build_model


def test_build_model():
    # arrange
    model = build_model()

    # act
    print(str(model))

    # assert
    assert 'RandomForestClassifier' in str(model)
    

def test_train_model():
    pass


def test_test_model():
    pass


def test_predict():
    pass