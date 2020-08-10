""" Train a classifier model and predict using a model. 

This module contains the following functions:
    train_model(): An interface to train a binary linear classifier model based
        on user's choices (e.g. use lr or svm algorithm, run private or 
        non-private version). 
    predict_linearmodel(): Predict labels y for the input features X using the 
        input binary linear classifier.  
"""
import numpy as np

from .dp_svm import dp_svm
from .dp_lr import dp_lr


def train_model(X, y, input, site="local"):
    """Trains a model depicted in input upon feature X and label y. 

    Args:
        X (ndarray of shape (n_sample, n_feature)): Features.
        y (ndarray of shape (n_sample,)): Labels.
        input (dict): Input of COINSTAC pipeline at each iteration.
        site (str): 'local' or 'owner'. Defaults to 'local'.

    Returns:
        w (ndarray of shape (n_feature,)): Weights in w'x.

    Raises:
        Exception: If site is neither 'owner' nor 'local'.
    """
    if site != "owner" and site != "local":
        raise Exception("Error in train_model: wrong site name")

    if input["model_" + site] == "LR":
        w = dp_lr(
            X,
            y,
            is_private=input["is_private_" + site],
            perturb_method=input["perturb_method_" + site],
            lambda_=input["lambda_" + site],
            epsilon=input["epsilon_" + site],
        )
    else:
        w = dp_svm(
            X,
            y,
            is_private=input["is_private_" + site],
            perturb_method=input["perturb_method_" + site],
            lambda_=input["lambda_" + site],
            epsilon=input["epsilon_" + site],
            huberconst=input["huberconst_" + site],
        )
    return w


def predict_linearmodel(weights, intercept, X):
    """Returns predicted labels for feature matrix X.

    Args:
        weights (ndarray of shape (n_feature,)): Weights in w'x + intercept.
        intercept: float, intercept in the model y = w'x + intercept, 
            for unpreprocessed raw data.
        X (ndarray of shape (n_sample, n_feature)):  Features.

    Returns:
        int ndarray of shape (n_sample,): Predicted labels (-/+1).
    """
    return np.where((np.matmul(X, weights) + intercept) >= 0, 1, -1)