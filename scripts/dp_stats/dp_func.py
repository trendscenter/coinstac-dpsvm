import numpy as np


def noisevector(dim, rate_lambda):
    """Generate vector-valued noise (dimension d) following Laplace distribution."""
    # https://ergodicity.net/2013/03/21/generating-vector-valued-noise-for-differential-privacy/
    # 1. The distribution of norm is Erlang distribution with parameters (dim, rate_lambda).
    # 2. For the direction, we can pick uniformly by sampling d i.i.d. Gaussians and normalizing them.
    
    # generate norm
    normn = np.random.default_rng().gamma(dim, 1 / rate_lambda, 1)  # after Numpy version 1.17.0
    # generate direction
    r1 = np.random.normal(0, 1, dim)
    n1 = np.linalg.norm(r1, 2)  # get the norm of r1
    r2 = r1 / n1  # normalize r1
    # get the result noise vector
    res = r2 * normn 
    return res