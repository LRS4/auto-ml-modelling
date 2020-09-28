# Automated Machine Learning Modelling

# Background

This project aims to provide a structure for delivering automated machine learning models and pipelines. The code will be modular and therefore easier to change for future production deployments. Some of the features include:

* Modular code based upon single responsibiliy principle
* Automated machine learning for model discovery
* Automated pipelines for repeatability
* Storing the final model in binary format using pickle

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

# References

* [Maintainable code in Data Science](https://github.com/klemag/pydataLDN_2019-maintainable-code-for-data-science)
* [Using TPOT](https://www.datacamp.com/community/tutorials/tpot-machine-learning-python)
