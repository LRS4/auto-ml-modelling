DATA_FILENAME = 'titanic.csv'
DATA_DIR = '../data/'
MODEL_NAME = 'model'
X_TRAIN = 'X_train.csv'
Y_TRAIN = 'y_train.csv'
X_TEST = 'X_test.csv'
Y_TEST = 'y_test.csv'
TARGET_COLUMN_NAME = 'Survived'

MAPPINGS = {
    'TARGET_DECODE': {0: 'Non-survivor', 1: 'Survivor'},
    'EMBARKED_NAMES': {'S': 'Southampton','Q': 'Queenstown', 'C': 'Cherbourg'},
    'EMBARKED': {'S': 1,'Q': 2, 'C': 3},
    'SEX': {'male': 1, 'female': 2}
}

PARAMS = {
    'test_size': 0.2,
    'features_subset': ["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]
}