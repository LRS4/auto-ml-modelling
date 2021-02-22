import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

## TITANIC PIPELINE

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.7156323644933228
exported_pipeline = MLPClassifier(alpha=0.01, learning_rate_init=0.01)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)


## IRIS PIPELINE


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import Normalizer
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive
from sklearn.preprocessing import FunctionTransformer
from copy import copy

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('https://gist.githubusercontent.com/netj/8836201/raw/6f9306ad21398ea43cba4f7d537619d0e07d5ae3/iris.csv')
tpot_data = tpot_data.rename(columns={'variety': 'target'})
tpot_data['target'] = tpot_data['target'].map({'Setosa': 0, 'Versicolor': 1, 'Virginica': 2})

features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.9833333333333332
exported_pipeline = make_pipeline(
    make_union(
        Normalizer(norm="l2"),
        FunctionTransformer(copy)
    ),
    GaussianNB()
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
score = exported_pipeline.score(testing_features, testing_target)
print(f'Score: {score}')

decode_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
custom_values = [[5.1, 3.5, 1.4, 0.2], [6.6, 3.0, 4.6, 1.4], [6.4, 2.8, 5.6, 2.1]]
custom_predictions = pd.Series(exported_pipeline.predict(custom_values)).map(decode_mapping)
custom_probabilities = exported_pipeline.predict_proba(custom_values)
custom_probabilities = pd.Series([max(arr) for arr in custom_probabilities])

np.set_printoptions(suppress=True)
df = pd.DataFrame(columns = ['prediction', 'probability %'])
df['prediction'], df['probability %'] = custom_predictions, custom_probabilities
print(df)


