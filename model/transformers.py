import os
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DATA_DIR, DATA_FILENAME,
                    TARGET_COLUMN_NAME, X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, PARAMS, MAPPINGS)


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
            df.to_csv(path_or_buf=f'{DATA_DIR}{key}.csv', index=False)
        print(f'Splits saved successfully in {DATA_DIR} folder')


    def print_summary(self, splits: dict):
        print('Printing summary...')
        print(f'Complete dataset: {self.X.shape}')
        for key in splits:
            df = splits[key]
            print(f'{key}: {df.shape}')


    def split_dataset(self, save_splits=True, print_summary=True):
        print(f'Splitting dataset into train and test sets. Ratio is ' + 
            str(int(100 - (PARAMS['test_size'] * 100))) + 
            ' / ' + 
            str(int(PARAMS['test_size'] * 100)))

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


class TrainDataPreprocessor():
    """ Preprocesses the training dataset """

    def __init__(self):
        self.X_train, self.y_train = self.load_training_datasets()

    
    def load_training_datasets(self):
        X_train = pd.read_csv(os.sep.join([DATA_DIR, X_TRAIN]))
        y_train = pd.read_csv(os.sep.join([DATA_DIR, Y_TRAIN]))
        
        return X_train[PARAMS['features_subset']], y_train


    def get_prepared_training_data(self):
        self.impute_missing_values()
        self.map_categoric_columns_to_numeric()

        return self.X_train.values, self.y_train.values.ravel()

    
    def impute_missing_values(self):
        self.X_train['Age'] = self.X_train['Age'].fillna(self.X_train['Age'].mean())
        self.X_train['Fare'] = self.X_train['Fare'].fillna(self.X_train['Fare'].median())
        self.X_train['Embarked'] = self.X_train['Embarked'].fillna('S')

    
    def map_categoric_columns_to_numeric(self):
        self.X_train['Sex'] = self.X_train['Sex'].map(MAPPINGS['SEX'])
        self.X_train['Embarked'] = self.X_train['Embarked'].map(MAPPINGS['EMBARKED'])
    

class TestDataPreprocessor():
    """ Preprocesses the test dataset """
    
    def __init__(self):
        self.X_test, self.y_test = self.load_test_datasets()

    
    def load_test_datasets(self):
        X_test = pd.read_csv(os.sep.join([DATA_DIR, X_TEST]))
        y_test = pd.read_csv(os.sep.join([DATA_DIR, Y_TEST]))
        
        return X_test[PARAMS['features_subset']], y_test


    def get_prepared_test_data(self):
        self.impute_missing_values()
        self.map_categoric_columns_to_numeric()

        return self.X_test.values, self.y_test.values.ravel()
    

    def impute_missing_values(self):
        self.X_test['Age'] = self.X_test['Age'].fillna(self.X_test['Age'].mean())
        self.X_test['Fare'] = self.X_test['Fare'].fillna('S')
        self.X_test['Embarked'] = self.X_test['Embarked'].fillna('S')


    def map_categoric_columns_to_numeric(self):
        self.X_test['Sex'] = self.X_test['Sex'].map(MAPPINGS['SEX'])
        self.X_test['Embarked'] = self.X_test['Embarked'].map(MAPPINGS['EMBARKED'])