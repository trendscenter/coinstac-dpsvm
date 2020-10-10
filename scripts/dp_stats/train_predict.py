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


def predict_proba_lr(weights, intercept, X):
    """Returns predicted probability for feature matrix X by logistic regression.

    Args:
        weights (ndarray of shape (n_feature,)): Weights in w'x + intercept.
        intercept: float, intercept in the model y = w'x + intercept, 
            for unpreprocessed raw data.
        X (ndarray of shape (n_sample, n_feature)): Features.

    Returns:
        float ndarray of shape (n_sample,): predicted probalility 
            by logistic regression, 1 / (1 + exp(-w'X)), range [0, 1].
    """
    n_sample = X.shape[0]
    wX = np.matmul(X, weights) + intercept * np.ones((n_sample,))  # w'X
    proba = np.zeros((n_sample,))
    # prevent overflow in exp()
    for i in range(n_sample):
        z = wX[i]
        if z >= 0:
            proba[i] = 1 / (1 + np.exp(-z))
        else:
            proba[i] = np.exp(z) / (np.exp(z) + 1)
    return proba


def predict_decision_svmhuber(weights, intercept, X):
    """Returns decision values for feature matrix X by SVM with huber loss.

    Args:
        weights (ndarray of shape (n_feature,)): Weights in w'x + intercept.
        intercept: float, intercept in the model y = w'x + intercept, 
            for unpreprocessed raw data.
        X (ndarray of shape (n_sample, n_feature)): Features.

    Returns:
        float ndarray of shape (n_sample,): w'x + intercept. 
        Used as y_score in sklearn.metrics.roc_auc_score.
    """
    return np.matmul(X, weights) + intercept
