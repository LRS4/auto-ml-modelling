import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer

from config import (DATA_DIR, DATA_FILENAME,
                    TARGET_COLUMN_NAME, PARAMS, MAPPINGS)


class ColumnsSelector(TransformerMixin, BaseEstimator):
    """ Selects chosen columns as features from dataframe """

    def __init__(self, column_names=[]):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.loc[:, self.column_names]


class MissingValuesImputer(TransformerMixin, BaseEstimator):
    """ Impute NaN values in given columns """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df.fill_na(inplace=True)


class ColumnsRemover(TransformerMixin, BaseEstimator):
    """ Drops columns from the features dataframe """

    def __init__(self, column_names=[]):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.drop(self.column_names, axis=1)


class TitleExtractor(TransformerMixin, BaseEstimator):
    """ Creates a new category of title """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame({
            'Title': X['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0]).apply(lambda x: x.replace('.', '')),
        })


class FamilySizeExtractor(TransformerMixin, BaseEstimator):
    """ Extracts FamilySize feature from the data """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame({
            'FamilySize': X['Parch'] + X['SibSp'] + 1
        })


class HasCabinExtractor(TransformerMixin, BaseEstimator):
    """ Extracts hasCabin feature from the data """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame({
            'hasCabin': X['Cabin'].notnull().astype(int),
        })


class EmbarkedAdjuster(TransformerMixin, BaseEstimator):
    """ Maps names of embarkation to initials """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return pd.DataFrame({
            'Embarked': X['Embarked'].map(MAPPINGS['EMBARKED'])
        })


class TrainTestSplitter():
    """ Splits the complete dataset into train/test sets """

    def __init__(self):
        self.X, self.y = self.load_full_dataset()

    def load_full_dataset(self):
        """ Loads the complete dataset returning X (features) and y (target) """
        dataset = pd.read_csv(os.sep.join([DATA_DIR, DATA_FILENAME]))
        X, y = dataset.drop(TARGET_COLUMN_NAME,
                            axis=1), dataset[TARGET_COLUMN_NAME]
        return X, y

    def save_splits_to_csv(self, splits: dict):
        for key in splits:
            df = splits[key]
            df.to_csv(path_or_buf=f'{DATA_DIR}{key}.csv')
        print(f'Splits saved successfully in {DATA_DIR} folder')

    def print_summary(self, splits: dict):
        print('Printing summary...')
        print(f'Complete dataset: {self.X.shape}')
        for key in splits:
            df = splits[key]
            print(f'{key}: {df.shape}')

    def split_dataset(self, save_splits=True, print_summary=True):
        print('Splitting dataset into test/train...')
        X, y = self.load_full_dataset()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=PARAMS['test_size'], random_state=42, stratify=y)
        splits = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

        if save_splits:
            self.save_splits_to_csv(splits)

        if print_summary:
            self.print_summary(splits)

        return splits
