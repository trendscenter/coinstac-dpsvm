"""General functions required in differentially private algorithms.
"""
import numpy as np


def noisevector(dim, rate_lambda):
    """Generates a noise vector following Laplace distribution.

    The distribution of norm is Erlang distribution with parameters (dim, 
    rate_lambda). For the direction, pick uniformly by sampling dim number of 
    i.i.d. Gaussians and normalizing them.

    Args:
        dim (int): Dimension of the noise vector.
        rate_lambda (float): epsion^(-rate_lambda*x) in Erlang distribution.
    
    Returns:
        res (ndarray of shape (dim,)): Noise vector.
    
    References:
        https://ergodicity.net/2013/03/21/generating-vector-valued-noise-for-differential-privacy/
    """
    # generate norm, after Numpy version 1.17.0
    normn = np.random.default_rng().gamma(dim, 1 / rate_lambda, 1)
    # generate direction
    r1 = np.random.normal(0, 1, dim)
    n1 = np.linalg.norm(r1, 2)  # get the norm of r1
    r2 = r1 / n1  # normalize r1
    # get the result noise vector
    res = r2 * normn
    return res
