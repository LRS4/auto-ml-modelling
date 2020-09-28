from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

iris = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
iris = iris.rename(columns={'variety': 'class'})
iris['class'] = iris['class'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

print(f"All data: {iris.shape}")

X_train, X_test, y_train, y_test = train_test_split(iris.drop(['class'], axis=1).astype(np.float64),
    iris['class'].astype(np.float64), train_size=0.8, test_size=0.2, random_state=42)

print(f'X_train: {X_train.shape}')
print(f'y_train: {y_train.shape}')
print(f'X_test: {X_test.shape}')
print(f'y_test: {y_test.shape}')

tpot = TPOTClassifier(generations=5,
                      population_size=50,
                      verbosity=2, 
                      random_state=42, 
                      max_time_mins=None)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))
tpot.export('tpot_iris_pipeline.py') 