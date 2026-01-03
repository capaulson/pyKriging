"""
Utility Functions for pyKriging

This module provides helper functions for:
- Model persistence (save/load)
- Data normalization
- Cross-validation data splitting
- Error metrics

Author: chrispaulson
"""

__author__ = 'chrispaulson'
import dill as pickle
import numpy as np
from copy import deepcopy


def norm(x):
    """
    Normalize array to range [0, 1].

    Args:
        x: Input array

    Returns:
        Normalized array with values in [0, 1]
    """
    x =  ((x)/(max(x)-min(x)))
    return x-min(x)


def saveModel(model, filePath):
    """
    Save a trained Kriging model to disk.

    Uses dill (extended pickle) for serialization, which handles
    lambda functions and closures that standard pickle cannot.

    Args:
        model: Trained kriging model object
        filePath: Path to save the model file

    Note:
        Models can be large due to stored matrices.
        Consider saving only hyperparameters for very large models.
    """
    pickle.dump(model, open(filePath, 'w'), byref=True)


def loadModel(filePath):
    """
    Load a saved Kriging model from disk.

    Args:
        filePath: Path to the saved model file

    Returns:
        Loaded kriging model object, ready for predictions
    """
    return pickle.load(open(filePath,'r'))


def splitArrays(krigeModel, q=5):
    """
    Split model data into train/test sets for cross-validation.

    Implements q-fold cross-validation by dividing the data into
    q approximately equal parts. Each iteration yields one part
    as test data and the remaining parts as training data.

    Args:
        krigeModel: Kriging model with X and y attributes
        q: Number of folds (default 5)

    Yields:
        Tuple of (trainX, trainy, testX, testy) for each fold
    """
    ind = np.arange(krigeModel.n)
    np.random.shuffle(ind)
    test = np.array_split(ind,q)
    for i in test:
        newX = deepcopy(krigeModel.X)
        newy = deepcopy(krigeModel.y)

        testX = newX[i]
        testy = newy[i]

        trainX = np.delete(newX,i,axis=0)
        trainy = np.delete(newy,i,axis=0)
        yield trainX, trainy, testX, testy


def mse(actual, predicted):
    """
    Calculate Mean Squared Error between actual and predicted values.

    Args:
        actual: True/observed value(s)
        predicted: Model predicted value(s)

    Returns:
        Squared error (actual - predicted)^2
    """
    return ((actual - predicted) ** 2)
