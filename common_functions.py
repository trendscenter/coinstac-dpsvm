from scipy import sign, dot
from matplotlib.pylab import demean


def test_errors(ws, data, labels):
    error = 100 * abs(
        sum(map(lambda x: min(0, x), sign(
            dot(ws, data.T)) * labels))) / labels.shape[0]
    return error


def data2data(data, clfs):
    """
    Applies all classifiers to the data and returns their output as vectors
    """
    X = dot(data, clfs)
    X = demean(X, axis=1)
    return X
