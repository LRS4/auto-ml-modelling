from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd


def main():
    """
    Uses tpot (Tree-based Pipeline Optimization Tool) an Automated Machine Learning tool 
    to find and output the best machine learning model for the given dataset. 
    
    See https://github.com/EpistasisLab/tpot

    Outputs the results to automodel.py
    """
    titanic = pd.read_csv('../data/titanic.csv')
    titanic.rename(columns={'Survived': 'class'}, inplace=True)

    for category in ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:
        print("Number of levels in category '{0}': \b {1:2.2f} ".format(category, titanic[category].unique().size))

    # Encode values
    titanic['Sex'] = titanic['Sex'].map({'male':0,'female':1})
    titanic['Embarked'] = titanic['Embarked'].map({'S':0,'C':1,'Q':2})

    # Fill na
    titanic = titanic.fillna(-999)
    pd.isnull(titanic).any()

    # Encode values
    from sklearn.preprocessing import MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    CabinTrans = mlb.fit_transform([{str(val)} for val in titanic['Cabin'].values])

    # Drop unused columns
    titanic_new = titanic.drop(['PassengerId', 'Name','Ticket','Cabin','class'], axis=1)

    # Create numpy arrays
    titanic_new = np.hstack((titanic_new.values,CabinTrans))
    titanic_class = titanic['class'].values

    # Train test split 
    # https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html
    # https://stackoverflow.com/questions/55525195/do-i-have-to-do-one-hot-encoding-separately-for-train-and-test-dataset
    training_indices, validation_indices = training_indices, testing_indices = train_test_split(titanic.index, stratify = titanic_class, train_size=0.75, test_size=0.25)
    training_indices.size, validation_indices.size

    # Train model
    tpot = TPOTClassifier(verbosity=2, max_time_mins=2, max_eval_time_mins=0.04, population_size=40)
    tpot.fit(titanic_new[training_indices], titanic_class[training_indices])

    # Score
    tpot.score(titanic_new[validation_indices], titanic.loc[validation_indices, 'class'].values)

    # Export
    tpot.export('automodel.py')


if __name__ == '__main__':
    main()