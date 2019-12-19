import numpy as np
from matplotlib.pylab import demean

from scipy import sign


def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v


def test_errors(ws, data, labels):
    error = 100 * abs(
        sum(map(lambda x: min(0, x),
                sign(np.dot(ws, data.T)) * labels))) / labels.shape[0]
    return error


def data2data(data, clfs):
    """
    Applies all classifiers to the data and returns their output as vectors
    """
    X = np.transpose([np.dot(data, clf) for clf in clfs])
    X = demean(X, axis=1)
    return X
