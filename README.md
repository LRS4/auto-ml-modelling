# Automated Machine Learning Modelling

# Background

This project aims to provide a modular structure for delivering automated machine learning models and pipelines. The code will be modular and therefore easier to change for future production deployments. Some of the features include:

* Modular code based upon single responsibiliy principle
* DRY principle, seperation of concerns, and easier to change principle
* Automated machine learning for model discovery
* Automated pipelines for repeatability
* Storing the final model in binary format using pickle
* Unit testing with pytest

This template uses the titanic dataset to keep things simple. However this structure could be adapted
and used for other projects as a starting point.

# Folder structure

```

├── data                   # Data CSV files including raw data and train test splits
├── model                  # Model logic and entry point run.py 
│   ├── config.py          # Holds volatile configuration values
│   ├── model.py           # Model for performing prediction
│   ├── run.py             # Entry point for the program
│   ├── tests.py           # Unit tests
│   ├── transformers.py    # Any preprocessing or transformation operations to the data
├── outputs                # Any outputted files such as model in binary format (.joblib) and predictions
├── utils                  # Helper files - tpot to provide automated model finding

```

# Install packages

The project uses [pipenv](https://pipenv-fork.readthedocs.io/en/latest/) to manage packages and dependencies. To install pipenv use:
```
pip install --user pipenv
```
Then add C:\Users\<Username>\AppData\Roaming\Python36\Scripts to path.

To install all package dependencies from the piplock file use:
```
pipenv install
```

# Example usage

To run tpot (Tree-based Pipeline Optimization Tool) an Automated Machine Learning tool 
to find and output the best machine learning model use:
```
cd utils
python model_finder.py
```
This outputs a file `automodel.py` which has a variable exported_pipeline showing
the most optimal model for the given data.


To split the main dataset into train/test sets use:
```
cd model
python run.py split
```
This creates a training set X_train (features) and y_train (target), alongside a test set X_test (live features) and y_test (live target answers) which should serve as unseen 'in the wild' data values. 


To train the model on the training dataset use:
```
cd model
python run.py train
```
This gathers prepared data using the TrainDataPreprocessor class and trains the model. It
also outputs accuracy metrics and cross validation scores. Finally, it outputs the model
to binary format in `..outputs/model.joblib`


To test the model on the test dataset (acts as unseen data) use:
```
cd model
python run.py test
```
This gathers prepared data using the TestDataPreprocessor class and outputs prediction accuracy.


To output predictions on the test dataset (acts as unseen data) use:
```
cd model
python run.py predict
```
This concatenates X_test, y_test and the newly created predictions for manual sense checking
within `../outputs/predictions.csv`. Configured to use X_test but this could be any new 'unseen' data
that needs predictions.


# Tests
To run all unit tests use (with optional -s for print output and -v for verbose):
```
cd model
pytest tests.py -s -v
```

# References

* [Maintainable code in Data Science](https://github.com/klemag/pydataLDN_2019-maintainable-code-for-data-science)
* [Using TPOT](https://www.datacamp.com/community/tutorials/tpot-machine-learning-python)
