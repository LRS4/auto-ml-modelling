import os
import joblib
import pandas as pd 
import sklearn.metrics as sklm
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from .transformers import TestDataPreprocessor, TrainDataPreprocessor
from .config import (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST, DATA_DIR,
                    DATA_FILENAME, TARGET_COLUMN_NAME, PARAMS)


def build_model():
    """
    Builds the model ready for training, the model used here
    is the one identified using tpot in model_finder.py
    """ 
    return RandomForestClassifier(bootstrap=True, 
                                  criterion="entropy", 
                                  max_features=0.7500000000000001, 
                                  min_samples_leaf=6, 
                                  min_samples_split=10, 
                                  n_estimators=100)


def train_model():
    """
    Applies the built model to X_train y_train sets.
    Outputs accuracy, cross validation metrics and 
    stores model in a joblib file for deployment.
    """
    preprocessor = TrainDataPreprocessor()
    X_train, y_train = preprocessor.get_prepared_training_data()

    model = build_model()
    model.fit(X_train, y_train)
    score = model.score(X_train, y_train)

    joblib.dump(model, '../outputs/model.joblib')
    print('Model saved to ../outputs/model.joblib', end="\n\n")

    print(f'Model training accuracy: { round(score * 100, 2) } %', end="\n\n")

    print('Performing 10 fold cross validation...')
    model_cross_validation_score = cross_val_score(model, X_train, y_train, cv=10, scoring='accuracy')
    print(f'10 fold cross validation accuracy: { round(model_cross_validation_score.mean() * 100, 2) } %', end='\n\n')

    print('Creating confusion matrix...')
    y_train_predictions = cross_val_predict(model, X_train, y_train, cv=10)
    confusion_matrix = sklm.confusion_matrix(y_train, y_train_predictions)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % confusion_matrix[0,0] + '             %5d' % confusion_matrix[0,1])
    print('Actual negative    %6d' % confusion_matrix[1,0] + '             %5d' % confusion_matrix[1,1])
    print('')
    print(f'Count of incorrectly classified  { confusion_matrix[1,0] + confusion_matrix[0,1] } / { len(y_train) }')
    print(f'Accuracy  { round(sklm.accuracy_score(y_train, y_train_predictions) * 100, 2) } %')
    print(' ')


def test_model():
    """
    Tests the trained model against X_test y_test sets.
    Outputs test accuracy evaluation.
    """
    try:
        model = joblib.load('../outputs/model.joblib')
    except:
        print('Must run python run.py train first to \
               create model.joblib binary file')

    preprocessor = TestDataPreprocessor()
    X_test, y_test = preprocessor.get_prepared_test_data()

    score = model.score(X_test, y_test)

    print(f'Model testing accuracy: { round(score * 100, 2) } %')


def predict():
    """
    Uses the model to predict on new data.
    For now this is set to y_test as this represents
    'unseen' data. Outputs X_test, y_test and predictions
    in a csv for manual sense checking
    """
    try:
        model = joblib.load('../outputs/model.joblib')
    except:
        print('Must run python run.py train first to \
               create model.joblib binary file')

    preprocessor = TestDataPreprocessor()
    X_test, y_test = preprocessor.get_prepared_test_data()

    print('Creating predictions against X_test (unseen data)')
    predictions = model.predict(X_test)

    X_test_csv = pd.read_csv(os.sep.join([DATA_DIR, X_TEST]))
    X_test_csv['Survived'] = y_test
    X_test_csv['Survived_PREDICTION'] = predictions

    X_test_csv.to_csv('../outputs/predictions.csv', index=False)
    print('Predictions saved to ../output/predictions.csv', end='\n\n')

    print('Creating confusion matrix for test predictions...')
    confusion_matrix = sklm.confusion_matrix(y_test, predictions)
    print('                 Confusion matrix')
    print('                 Score positive    Score negative')
    print('Actual positive    %6d' % confusion_matrix[0,0] + '             %5d' % confusion_matrix[0,1])
    print('Actual negative    %6d' % confusion_matrix[1,0] + '             %5d' % confusion_matrix[1,1])
    print('')
    print(f'Count of incorrectly classified  { confusion_matrix[1,0] + confusion_matrix[0,1] } / { len(y_test) }')
    print(f'Accuracy  { round(sklm.accuracy_score(y_test, predictions) * 100, 2) } %')
    print(' ')
