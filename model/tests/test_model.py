import pytest
import os
import pandas as pd

from model.model import build_model


def test_build_model():
    # arrange
    build_model()

    # act

    # assert
    assert 1 == 1