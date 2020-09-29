import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from transformers import ColumnsSelector
from pipeline import exported_pipeline as classifier

from config import (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, DATA_DIR,
                    DATA_FILENAME, TARGET_COLUMN_NAME, PARAMS)


def load_dataset(x_path, y_path):
    """
    Loads  X (features) and y (targets) for train or test sets
    """
    X = pd.read_csv(os.sep.join([DATA_DIR, x_path]))
    y = pd.read_csv(os.sep.join([DATA_DIR, y_path]))
    return X, y


def build_model():
    pass


def train_model():
    pass


def tune_pipeline():
    pass
