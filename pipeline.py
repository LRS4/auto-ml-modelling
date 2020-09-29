import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Average CV score on the training set was: 0.8383121984064639
exported_pipeline = RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.8, min_samples_leaf=3, min_samples_split=6, n_estimators=100)
