"""Differentially private regularized support vector machine (svm).

This module exports functions to train a non-private / differentially private 
regularized binary svm classifier. For the differentially private version, 
output or objective perturbation can be used. 

Usage example:
    w = dp_svm(
        X, y, 
        is_private=False,
        perturb_method='output',
        lambda_=0.01, 
        epsilon=0.1, 
        huberconst=0.5
    )    

    w (ndarray of shape (n_feature,)): Weights in w'x in svm classifier.
"""
import numpy as np
from scipy.optimize import minimize

from .general_funcs import noisevector


def huberloss(z, huberconst):
    """Returns normal Huber loss (float) for a sample. 

    Args:
        z (float): x_i (1, n_feature) * y_i (int: -/+1) * w (n_feature, 1).
        huberconst (float): Huber loss parameter.        

    References:
        chaudhuri2011differentially: equation 7 & corollary 13
    """
    if z > 1.0 + huberconst:
        hloss = 0
    elif z < 1.0 - huberconst:
        hloss = 1 - z
    else:
        hloss = (1 + huberconst - z) ** 2 / (4 * huberconst)
    return hloss


def eval_svm(weights, XY, num, lambda_, b, huberconst):
    """Evaluates differentially private regularized svm loss for a data set. 

    Args:
        weights (ndarray of shape (n_feature,)): Weights in w'x in classifier.
        XY (ndarray of shape (n_sample, n_feature)): 
            each row x_i (1, n_feature) * label y_i (int: -/+1).
        num (int): n_sample.
        lambda_ (float): Regularization parameter. 
        b (ndarray of shape (n_feature,)): Noise vector. If b is a zero vector,
            returns non-private regularized svm loss.
        huberconst (float): Huber loss parameter.
    
    Returns:
        fw (float): Differentially private regularized svm loss.
    """
    # add Huber loss from all samples
    XYW = np.matmul(XY, weights)
    fw = np.mean(
        [huberloss(z=z, huberconst=huberconst) for z in XYW], dtype=np.float64
    )
    # add regularization term (1/2 * lambda * |w|^2) and b term (1/n * b'w)
    fw += 0.5 * lambda_ * weights.dot(weights) + 1.0 / num * b.dot(weights)
    return fw


def train_svm_nonpriv(XY, num, dim, lambda_, huberconst):
    """Trains a non-private regularized svm classifier. 

    Args:
        XY (ndarray of shape (n_sample, n_feature)): 
            each row x_i (1, n_feature) * label y_i (int: -/+1).
        num (int): n_sample.
        dim (int): Dimension of x_i, i.e., n_feature.
        lambda_ (float): Regularization parameter. 
        huberconst (float): Huber loss parameter.
    
    Returns:
        w_nonpriv (ndarray of shape (n_feature,)): 
            Weights in w'x in svm classifier.

    Raises:
        Exception: If the optimizer exits unsuccessfully.
    """
    w0 = np.zeros(dim)  # w starting point
    b = np.zeros(dim)  # zero noise vector
    res = minimize(
        eval_svm,
        w0,
        args=(XY, num, lambda_, b, huberconst),
        method="L-BFGS-B",
        bounds=None,
    )
    if not res.success:
        raise Exception(res.message)
    w_nonpriv = res.x
    return w_nonpriv


def train_svm_outputperturb(XY, num, dim, lambda_, epsilon, huberconst):
    """Trains a private regularized svm classifier by output perturbation. 

    First, train a non-private svm classifier to get weights w_nonpriv, 
    then add noise to w_nonpriv to get w_priv.

    Args:
        XY (ndarray of shape (n_sample, n_feature)): 
            each row x_i (1, n_feature) * label y_i (int: -/+1).
        num (int): n_sample.
        dim (int): Dimension of x_i, i.e., n_feature.
        lambda_ (float): Regularization parameter. 
        epsilon (float): Privacy parameter.
        huberconst (float): Huber loss parameter.
    
    Returns:
        w_priv (ndarray of shape (n_feature,)): 
            Weights in w'x in svm classifier.
    
    References:
        chaudhuri2011differentially: Algorithm 1 output perturbation.
    """
    w_nonpriv = train_svm_nonpriv(
        XY=XY, num=num, dim=dim, lambda_=lambda_, huberconst=huberconst
    )
    beta = num * lambda_ * epsilon / 2
    noise = noisevector(dim, beta)
    w_priv = w_nonpriv + noise
    return w_priv


def train_svm_objectiveperturb(XY, num, dim, lambda_, epsilon, huberconst):
    """Trains a private regularized svm classifier by objective perturbation. 

    Add noise to the objective (loss function).

    Args:
        XY (ndarray of shape (n_sample, n_feature)): 
            each row x_i (1, n_feature) * label y_i (int: -/+1).
        num (int): n_sample.
        dim (int): Dimension of x_i, i.e., n_feature.
        lambda_ (float): Regularization parameter. 
        epsilon (float): Privacy parameter.
        huberconst (float): Huber loss parameter.
    
    Returns:
        w_priv (ndarray of shape (n_feature,)): 
            Weights in w'x in svm classifier.

    Raises:
        Exception: If epsilon_p < 1e-4, where epsilon_p is calculated from 
            n_sample, lambda_ and epsilon.
        Exception: If the optimizer exits unsuccessfully.
    
    References:
        chaudhuri2011differentially: Algorithm 2 objective perturbation.
        http://cseweb.ucsd.edu/~kamalika/code/dperm/documentation.pdf
    """
    c = 1 / (2 * huberconst)  # value for svm
    tmp = c / (num * lambda_)
    epsilon_p = epsilon - np.log(1.0 + 2 * tmp + tmp * tmp)
    if epsilon_p < 1e-4:
        raise Exception(
            "Error: Cannot run algorithm"
            + "for this n_sample, lambda, epsilon and huberconst value"
        )

    w0 = np.zeros(dim)
    beta = epsilon_p / 2
    b = noisevector(dim, beta)
    
    '''
    res = minimize(
        eval_svm,
        w0,
        args=(XY, num, lambda_, b, huberconst),
        method="L-BFGS-B",
        bounds=None,
    )
    From https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html#optimize-minimize-lbfgsb, it says maxfun: int
    Maximum number of function evaluations. Note that this function may violate the limit because of evaluating gradients by numerical differentiation. 
    Edited the minimize function and add maxfun argument
    '''
    
    res = minimize(
        eval_svm,
        w0,
        args=(XY, num, lambda_, b, huberconst),
        method="L-BFGS-B",
        bounds=None,
        options={'maxfun':np.inf}
    )
    
    if not res.success:
        raise Exception(res.message)
    w_priv = res.x
    return w_priv


def dp_svm(
    features,
    labels,
    is_private=True,
    perturb_method="objective",
    lambda_=0.01,
    epsilon=0.1,
    huberconst=0.5,
):
    """Trains a non-private or differentially private svm classifier. 

    Args:
        features (ndarray of shape (n_sample, n_feature)): X.
        labels (ndarray of shape (n_sample,)): y.
        is_private (bool): run private version or not. Defaults to True.
        perturb_method (str): Use output perturbation for 'output', objective 
            perturbation otherwise. Defaults to 'objective'.
        lambda_ (float): Regularization parameter. Defaults to 0.01.
        epsilon (float): Privacy parameter. Defaults to 0.1.
        huberconst (float): Huber loss parameter. Defaults to 0.5.

    Returns:
        w_nonpriv / w_priv (ndarray of shape (n_feature,)): 
            Weights in w'x in svm classifier.

    Raises:
        Exception: If lambda_ < 0 or huberconst < 0 for non-private version. If 
            lambda_ < 0 or huberconst < 0 or epsilon < 0 for private version.

    References:
        [1] K. Chaudhuri, C. Monteleoni, and A. D. Sarwate, 
        “Differentially  privateempirical risk minimization,” 
        Journal of Machine Learning Research, vol. 12,no. Mar, 
        pp. 1069–1109, 2011.
        [2] ——, “Documentation for regularized lr and regularized svm 
        code,” Available at 
        http://cseweb.ucsd.edu/kamalika/code/dperm/documentation.pdf.
    """
    num = features.shape[0]  # number of samples
    dim = features.shape[1]  # dimension of a sample vector x_i
    # svm function only needs [vector x_i * label y_i]
    XY = features * labels[:, np.newaxis]

    # non-private version
    if not is_private:
        if lambda_ < 0.0 or huberconst < 0.0:
            raise Exception(
                "ERROR: Lambda and Huberconst" + "should all be positive."
            )
        w_nonpriv = train_svm_nonpriv(
            XY=XY, num=num, dim=dim, lambda_=lambda_, huberconst=huberconst
        )
        return w_nonpriv
    # private version
    if lambda_ < 0.0 or epsilon < 0.0 or huberconst < 0.0:
        raise Exception(
            "ERROR: Lambda, Epsilon and Huberconst" + "should all be positive."
        )

    if perturb_method == "output":
        w_priv = train_svm_outputperturb(
            XY=XY,
            num=num,
            dim=dim,
            lambda_=lambda_,
            epsilon=epsilon,
            huberconst=huberconst,
        )
    else:
        w_priv = train_svm_objectiveperturb(
            XY=XY,
            num=num,
            dim=dim,
            lambda_=lambda_,
            epsilon=epsilon,
            huberconst=huberconst,
        )
    return w_priv
